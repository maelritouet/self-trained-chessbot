"""Decoder-only GPT over characters.

Interface contract (other code depends on this exactly):
    model = GPT(GPTConfig(vocab_size=33, block_size=1024, ...))
    logits, loss = model(idx)            # idx: (B, T) int64, T <= block_size
    logits, loss = model(idx, targets)   # targets: (B, T) int64 -> loss is a scalar
  - logits: (B, T, vocab_size) float, logits[b, t] predicts token t+1
  - loss:   None if targets is None, else mean cross-entropy over all B*T positions
  - model.config: the GPTConfig (player.py reads config.block_size)

TODO (suggested order, following Karpathy's "Let's build GPT"):
  [ ] token embedding (vocab_size -> n_embd) + learned position embedding (block_size -> n_embd)
  [ ] causal self-attention: q, k, v projections, n_head heads of size n_embd // n_head,
      scaled dot product, causal mask (no peeking at t' > t), softmax, output projection
  [ ] MLP: Linear(n_embd, 4*n_embd) -> GELU -> Linear(4*n_embd, n_embd)
  [ ] Block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))   (pre-LayerNorm, residuals)
  [ ] GPT: embeddings -> dropout -> n_layer Blocks -> final LayerNorm -> lm_head (n_embd -> vocab_size)
  [ ] loss: F.cross_entropy on logits reshaped to (B*T, vocab_size) and targets to (B*T,)
  [ ] num_params() helper; sanity-check against ~12 * n_layer * n_embd^2
  [ ] weight init (normal std 0.02); optional: tie lm_head and token embedding weights

Questions to be able to answer:
  - Why divide attention scores by sqrt(head_size)?
  - What breaks if the causal mask is missing, and how would the loss curve look?
  - Where do the ~12 * n_layer * n_embd^2 parameters come from?
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 33     # from meta.json
    block_size: int = 1024   # max context length (characters)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384        # must be divisible by n_head
    dropout: float = 0.0


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        raise NotImplementedError("model.py not implemented yet")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) int64 token ids, T <= config.block_size
            targets: (B, T) int64 next-token ids, or None
        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar tensor (mean cross-entropy) if targets is given, else None
        """
        raise NotImplementedError

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; non_embedding=True excludes the position embedding."""
        raise NotImplementedError
