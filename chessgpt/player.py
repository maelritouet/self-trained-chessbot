"""Turn a character-level model into a chess player: choose_move().

This is the single "move function" shared by the evals (W4), the Lichess bot, and
(ported to JS, with a parity test) the web demo (W6). It only relies on the GPT
interface:  logits, _ = model(idx)  with idx (B, T) -> logits (B, T, vocab_size),
and model.config.block_size.

How a move is chosen:
  1. Build the prompt exactly like a training game: "\n" + "1-0 " + "1.e4 e5 2."
     (delimiter, optional result prefix, movetext so far, next move number if white).
  2. Sample characters at `temperature` until a move boundary (space or delimiter).
  3. Check the sampled text is exactly the SAN of a legal move (legal_move_from_text).
     Anything else -> record it and
     resample, up to max_retries extra tries.
  4. Separately, score every legal move: P(move) = product of P(char) over its
     characters, times P(boundary after it). Return the top-k as candidates.

Usage:
    from chessgpt.player import choose_move, UniformModel
    choice = choose_move(model, tok, ["e4", "e5"], temperature=0.8, result_prefix="1-0")
    choice.move, choice.san, choice.candidates, choice.illegal_tries
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessgpt.movetext import format_movetext
from chessgpt.tokenizer import CharTokenizer

MAX_MOVE_CHARS = 10  # longest SAN is 7 ("Qa1xb2+" / "exd8=Q#"); give some slack


@dataclass
class Candidate:
    san: str
    move: chess.Move
    prob: float  # model probability of this exact move (at temperature 1)


@dataclass
class MoveChoice:
    move: chess.Move | None          # None if every try was illegal
    san: str | None
    candidates: list[Candidate]      # top-k legal moves by model probability, best first
    illegal_tries: list[str] = field(default_factory=list)  # raw strings the model sampled
    legal_mass: float = 0.0          # total probability on legal moves (1 - mass on illegal output)


def build_prompt(board_moves: list[str], result_prefix: str | None, delimiter: str = "\n") -> str:
    """Prompt in the training format, ending right where the next move's SAN starts.

    []           -> "\\n1-0 1."
    ["e4"]       -> "\\n1-0 1.e4 "
    ["e4", "e5"] -> "\\n1-0 1.e4 e5 2."
    """
    prompt = delimiter + (f"{result_prefix} " if result_prefix else "")
    if board_moves:
        prompt += format_movetext(board_moves) + " "
    if len(board_moves) % 2 == 0:  # white to move: next token is the move number
        prompt += f"{len(board_moves) // 2 + 1}."
    return prompt


def legal_move_from_text(board: chess.Board, text: str) -> chess.Move | None:
    """The legal move whose SAN is exactly `text`, else None.

    Strict on purpose: python-chess's parse_san is lenient (it accepts phantom captures
    like "Nxf3" for Nf3, missing or extra check marks, long algebraic, and "--" as a null
    move). Exact matching against the legal moves' SAN is the same rule in Python and in
    the JS port (chess.js), and the model was trained on exact SAN, so a wrong "x", "+" or
    "#" is a real mistake.
    """
    for move in board.legal_moves:
        if board.san(move) == text:
            return move
    return None


def replay(moves_so_far: list[str]) -> chess.Board:
    """Replay SAN moves from the start position (raises ValueError if any is illegal)."""
    board = chess.Board()
    for san in moves_so_far:
        board.push_san(san)
    return board


def _device(model: nn.Module) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _crop(ids: list[int], block_size: int) -> list[int]:
    # Long games exceed the context: keep the most recent block_size tokens.
    # (This drops the delimiter and result prefix, an out-of-distribution prompt;
    #  part of why legal-move rate is expected to fall late in long games.)
    return ids[-block_size:]


@torch.no_grad()
def sample_move_string(
    model: nn.Module, tok: CharTokenizer, prompt_ids: list[int], temperature: float,
    boundary_ids: set[int], generator: torch.Generator | None = None,
) -> str:
    """Sample characters until a boundary token; return them (without the boundary)."""
    device, block_size = _device(model), model.config.block_size
    ids, out = list(prompt_ids), []
    for _ in range(MAX_MOVE_CHARS):
        idx = torch.tensor([_crop(ids, block_size)], dtype=torch.long, device=device)
        logits, _ = model(idx)
        logits = logits[0, -1].float()
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            probs = F.softmax(logits / temperature, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        if nxt in boundary_ids:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


@torch.no_grad()
def score_legal_moves(
    model: nn.Module, tok: CharTokenizer, prompt_ids: list[int], board: chess.Board,
    boundary_ids: set[int],
) -> list[Candidate]:
    """Probability of every legal move, in one batched forward pass. Sorted best first.

    log P(move) = sum_i log P(c_i | prompt, c_<i) + log P(boundary | prompt, move)
    Sequences are right-padded to equal length; with a causal model, padding after
    a position can't affect that position's logits.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []
    sans = [board.san(m) for m in legal]
    device, block_size = _device(model), model.config.block_size

    seqs = [_crop(prompt_ids + tok.encode(s), block_size) for s in sans]
    T = max(len(s) for s in seqs)
    idx = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        idx[b, : len(s)] = torch.tensor(s)
    logits, _ = model(idx.to(device))
    logp = F.log_softmax(logits.float(), dim=-1).cpu()

    boundary = torch.tensor(sorted(boundary_ids))
    scores = []
    for b, (san, seq) in enumerate(zip(sans, seqs)):
        n = len(san)
        L = len(seq)
        # logits at position t predict token t+1: move chars sit at [L-n, L)
        char_lp = sum(float(logp[b, t - 1, seq[t]]) for t in range(L - n, L))
        end_lp = float(torch.logsumexp(logp[b, L - 1, boundary], dim=0))
        scores.append(char_lp + end_lp)

    cands = [Candidate(san, m, float(torch.tensor(s).exp())) for san, m, s in zip(sans, legal, scores)]
    return sorted(cands, key=lambda c: c.prob, reverse=True)


def choose_move(
    model: nn.Module,
    tokenizer: CharTokenizer,
    moves_so_far: list[str],
    temperature: float = 1.0,
    max_retries: int = 5,
    result_prefix: str | None = None,
    top_k: int = 5,
    delimiter: str = "\n",
    generator: torch.Generator | None = None,
) -> MoveChoice:
    """Pick the model's next move.

    Args:
        model:         anything with the GPT interface (see module docstring)
        tokenizer:     the CharTokenizer from the dataset's meta.json
        moves_so_far:  SAN moves from the start position, e.g. ["e4", "e5", "Nf3"]
        temperature:   sampling temperature; 0 = greedy (argmax)
        max_retries:   extra samples allowed after an illegal one (total tries = 1 + max_retries)
        result_prefix: "1-0" / "0-1" / "1/2-1/2" to condition on an outcome (the data has it
                       as a prefix, so pass the result you want the model's side to get).
                       None omits it, which is out-of-distribution for a prefix-trained model.
        top_k:         number of candidates returned
        delimiter:     game delimiter used in the dataset (meta.json "delimiter")
        generator:     optional torch.Generator for reproducible sampling

    Returns:
        MoveChoice. move is None if all tries were illegal; the caller decides what to
        do then (e.g. forfeit, or fall back to candidates[0]). Candidates are always
        computed at temperature 1, i.e. the model's actual distribution.
    """
    model.eval()
    board = replay(moves_so_far)
    if board.is_game_over():
        raise ValueError(f"game is already over: {board.result()}")

    prompt_ids = tokenizer.encode(build_prompt(moves_so_far, result_prefix, delimiter))
    boundary_ids = {tokenizer.stoi[c] for c in (" ", delimiter) if c in tokenizer.stoi}

    all_cands = score_legal_moves(model, tokenizer, prompt_ids, board, boundary_ids)
    choice = MoveChoice(
        move=None, san=None, candidates=all_cands[:top_k],
        legal_mass=sum(c.prob for c in all_cands),
    )

    for _ in range(1 + max_retries):
        s = sample_move_string(model, tokenizer, prompt_ids, temperature, boundary_ids, generator)
        move = legal_move_from_text(board, s)
        if move is None:  # illegal, not exact SAN, or empty
            choice.illegal_tries.append(s)
            continue
        choice.move, choice.san = move, s
        break
    return choice


class UniformModel(nn.Module):
    """Stand-in with the GPT interface: uniform logits over the vocab.

    Lets the player/eval harness run end to end before the real model exists.
    Expect nearly every sampled move to be illegal, and candidates ranked by SAN
    length (shorter = fewer factors of 1/vocab_size).
    """

    def __init__(self, vocab_size: int, block_size: int = 1024):
        super().__init__()
        from chessgpt.model import GPTConfig  # only the dataclass; GPT itself isn't needed

        self.config = GPTConfig(vocab_size=vocab_size, block_size=block_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.config.vocab_size, device=idx.device)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
