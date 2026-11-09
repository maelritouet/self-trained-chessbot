// JS port of chessgpt/player.py: turn the character-level model into a chess player.
//
// Environment-agnostic ES module: it imports nothing. The caller passes in
//   - runModel(rows) -> Promise<{data: Float32Array, dims: [B, T, V]}>, the GPT interface
//     (rows: B arrays of T token ids; logits[b, t] predicts token t+1), e.g. onnxRunner(ort, session);
//   - Chess, the chess.js class (the referee).
// The same file runs in the browser (onnxruntime-web) and in Node tests (onnxruntime-node),
// and web/test/player.test.mjs checks it against the Python version on fixed positions.
//
// How a move is chosen (identical to Python):
//   1. Prompt in the training format: "\n" + "1-0 " + "1.e4 e5 2." (delimiter, optional
//      result prefix, movetext so far, next move number if White is to move).
//   2. Sample characters at `temperature` until a boundary (space or delimiter).
//   3. Check the SAN with chess.js; if illegal, record it and resample (up to maxRetries more).
//   4. Separately, score every legal move: P(move) = prod P(char) * P(boundary after it).

const MAX_MOVE_CHARS = 10; // longest SAN is 7 ("Qa1xb2+", "exd8=Q#"); some slack

export function formatMovetext(sans) {
  return sans.map((san, i) => (i % 2 === 0 ? `${i / 2 + 1}.${san}` : san)).join(" ");
}

/** Prompt ending exactly where the next move's SAN starts. */
export function buildPrompt(moves, resultPrefix = null, delimiter = "\n") {
  let prompt = delimiter + (resultPrefix ? `${resultPrefix} ` : "");
  if (moves.length) prompt += formatMovetext(moves) + " ";
  if (moves.length % 2 === 0) prompt += `${moves.length / 2 + 1}.`;
  return prompt;
}

export class Tokenizer {
  constructor(itos) {
    this.itos = itos;
    this.stoi = new Map(itos.map((c, i) => [c, i]));
  }
  get vocabSize() { return this.itos.length; }
  encode(text) {
    return Array.from(text, (c) => {
      const id = this.stoi.get(c);
      if (id === undefined) throw new Error(`character ${JSON.stringify(c)} not in vocab`);
      return id;
    });
  }
  decode(ids) { return Array.from(ids, (i) => this.itos[Number(i)]).join(""); }
}

/** Adapt an onnxruntime InferenceSession (web or node) to the runModel interface. */
export function onnxRunner(ort, session) {
  return async (rows) => {
    const B = rows.length, T = rows[0].length;
    const idx = new ort.Tensor("int64", BigInt64Array.from(rows.flat(), (v) => BigInt(v)), [B, T]);
    const { logits } = await session.run({ idx });
    return { data: logits.data, dims: logits.dims };
  };
}

// --- maths helpers (double precision over the float32 logits) -----------------------------------
function logSoftmaxRow(data, offset, V) {
  let max = -Infinity;
  for (let v = 0; v < V; v++) max = Math.max(max, data[offset + v]);
  let sum = 0;
  for (let v = 0; v < V; v++) sum += Math.exp(data[offset + v] - max);
  const lse = max + Math.log(sum);
  const out = new Float64Array(V);
  for (let v = 0; v < V; v++) out[v] = data[offset + v] - lse;
  return out;
}

function logSumExp(values) {
  const max = Math.max(...values);
  return max + Math.log(values.reduce((s, v) => s + Math.exp(v - max), 0));
}

const crop = (ids, blockSize) => ids.slice(-blockSize); // keep the most recent tokens

/** Sample characters until a boundary token; return them (without the boundary). */
async function sampleMoveString(runModel, tok, promptIds, temperature, boundaryIds, blockSize, rng) {
  const ids = [...promptIds];
  const out = [];
  for (let step = 0; step < MAX_MOVE_CHARS; step++) {
    const row = crop(ids, blockSize);
    const { data, dims } = await runModel([row]);
    const [, T, V] = dims;
    const offset = (T - 1) * V; // last position of the only row
    let next;
    if (temperature <= 0) {
      next = 0; // argmax, first maximum on ties (same as torch.argmax)
      for (let v = 1; v < V; v++) if (data[offset + v] > data[offset + next]) next = v;
    } else {
      const logp = logSoftmaxRow(Array.from(data.subarray(offset, offset + V), (x) => x / temperature), 0, V);
      let u = rng(), acc = 0;
      next = V - 1;
      for (let v = 0; v < V; v++) {
        acc += Math.exp(logp[v]);
        if (u < acc) { next = v; break; }
      }
    }
    if (boundaryIds.has(next)) break;
    out.push(next);
    ids.push(next);
  }
  return tok.decode(out);
}

/** Probability of every legal move in one batched forward pass, best first.
 *  log P(move) = sum_i log P(c_i | prompt, c_<i) + log P(boundary | prompt, move).
 *  Rows are right-padded; with a causal model padding can't affect earlier positions. */
async function scoreLegalMoves(runModel, tok, promptIds, sans, boundaryIds, blockSize) {
  if (!sans.length) return [];
  const seqs = sans.map((san) => crop([...promptIds, ...tok.encode(san)], blockSize));
  const T = Math.max(...seqs.map((s) => s.length));
  const rows = seqs.map((s) => [...s, ...new Array(T - s.length).fill(0)]);
  const { data, dims } = await runModel(rows);
  const V = dims[2];
  const boundary = [...boundaryIds];
  return sans
    .map((san, b) => {
      const seq = seqs[b], L = seq.length, n = san.length;
      const at = (t) => logSoftmaxRow(data, (b * T + t) * V, V); // logits at t predict token t+1
      let lp = 0;
      for (let t = L - n; t < L; t++) lp += at(t - 1)[seq[t]];
      const last = at(L - 1);
      lp += logSumExp(boundary.map((id) => last[id]));
      return { san, prob: Math.exp(lp) };
    })
    .sort((a, b) => b.prob - a.prob);
}

/** `text` if it is exactly the SAN of a legal move, else null.
 *  Strict on purpose, same rule as chessgpt.player.legal_move_from_text: chess.js's move()
 *  (like python-chess's parse_san) is lenient and would accept "Nf3+" without check or
 *  long algebraic like "Ng1f3"; exact matching is identical in both languages. */
function legalSan(board, text) {
  return board.moves().includes(text) ? text : null;
}

/**
 * Pick the model's next move.
 * @returns {Promise<{san: string|null, candidates: {san: string, prob: number}[],
 *                    illegalTries: string[], legalMass: number}>}
 *   san is null if every try was illegal; the caller decides what to do then.
 *   Candidates are scored at temperature 1 (the model's actual distribution).
 */
export async function chooseMove({
  runModel, tokenizer, Chess, moves, temperature = 1.0, maxRetries = 5, resultPrefix = null,
  topK = 5, delimiter = "\n", blockSize, rng = Math.random,
}) {
  if (!blockSize) throw new Error("blockSize is required (meta.json block_size)");
  const board = new Chess();
  for (const san of moves) board.move(san);
  // Same rule as python-chess's board.is_game_over(): only endings that are automatic.
  // chess.js's isGameOver() also counts claimable draws (threefold, 50-move rule); whether
  // to claim those is the game loop's decision, not the move function's.
  if (board.isCheckmate() || board.isStalemate() || board.isInsufficientMaterial()) {
    throw new Error("game is already over");
  }

  const promptIds = tokenizer.encode(buildPrompt(moves, resultPrefix, delimiter));
  const boundaryIds = new Set([" ", delimiter].filter((c) => tokenizer.stoi.has(c)).map((c) => tokenizer.stoi.get(c)));

  const all = await scoreLegalMoves(runModel, tokenizer, promptIds, board.moves(), boundaryIds, blockSize);
  const choice = {
    san: null,
    candidates: all.slice(0, topK),
    illegalTries: [],
    legalMass: all.reduce((s, c) => s + c.prob, 0),
  };
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const text = await sampleMoveString(runModel, tokenizer, promptIds, temperature, boundaryIds, blockSize, rng);
    const san = legalSan(board, text);
    if (san === null) {
      choice.illegalTries.push(text);
      continue;
    }
    choice.san = san;
    break;
  }
  return choice;
}
