"""Decoder-only GPT over characters (GPT-2 / nanoGPT architecture).

Interface contract (other code depends on this exactly):
    model = GPT(GPTConfig(vocab_size=33, block_size=1024, ...))
    logits, loss = model(idx)            # idx: (B, T) int64, T <= block_size
    logits, loss = model(idx, targets)   # targets: (B, T) int64 -> loss is a scalar
  - logits: (B, T, vocab_size) float, logits[b, t] predicts token t+1
  - loss:   None if targets is None, else mean cross-entropy over all B*T positions
  - model.config: the GPTConfig (player.py reads config.block_size)

Map of the code (B = batch, T = sequence length, C = n_embd, H = n_head, D = C // H):
  idx (B,T) --token emb + pos emb--> x (B,T,C) --dropout-->
    n_layer x Block:  x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))
  --ln_f--> (B,T,C) --lm_head (tied to token emb)--> logits (B,T,V)

Questions to be able to answer:
  - Why divide attention scores by sqrt(D)?
  - What breaks if the causal mask is missing, and how would the loss curve look?
  - Where do the ~12 * n_layer * n_embd^2 parameters come from?
  - Why pre-LayerNorm (LN inside the residual branch) rather than after the addition?
  - Why scale the init of c_proj weights by 1/sqrt(2 * n_layer)?
  - What does weight tying save here, and why is it reasonable?
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 33     # from meta.json
    block_size: int = 1024   # max context length (characters)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384        # must be divisible by n_head
    dropout: float = 0.0
    # "sdpa": torch's fused scaled_dot_product_attention (fast, memory-efficient on GPU).
    # "manual": the same computation written out step by step (for reading/debugging).
    # Same parameters, so checkpoints work with either; a test checks they agree.
    attn_impl: str = "sdpa"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attn_impl = config.attn_impl
        # One matmul produces q, k and v for all heads: (C) -> (3C)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection: mixes the concatenated heads back into the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if config.attn_impl == "manual":
            # Lower-triangular mask: position t may attend to positions <= t only.
            mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size),
                                 persistent=False)  # not saved in checkpoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        D = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # 3 x (B, T, C)
        # Split channels into heads and put heads before time: (B, H, T, D)
        q = q.view(B, T, self.n_head, D).transpose(1, 2)
        k = k.view(B, T, self.n_head, D).transpose(1, 2)
        v = v.view(B, T, self.n_head, D).transpose(1, 2)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D)           # (B, H, T, T)
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)                              # rows sum to 1
            att = self.attn_dropout(att)
            y = att @ v                                               # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)              # re-join heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention communicates, the MLP computes."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # residual connections keep a clean gradient path
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})")
        if config.attn_impl not in ("sdpa", "manual"):
            raise ValueError(f"attn_impl must be 'sdpa' or 'manual', got {config.attn_impl!r}")
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned position embedding
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output layer reuses the token embedding matrix
        # (a character's input vector is also its output "prototype").
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2: each residual branch's output projection is scaled down so the
        # residual stream's variance doesn't grow with depth (2 branches per block).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.config.block_size}")

        pos = torch.arange(T, device=idx.device)                      # (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))                  # (B,T,C) + (T,C) broadcast
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                           # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; non_embedding=True excludes the position embedding.

        Tied weights are counted once (parameters() deduplicates them). The token
        embedding stays in the count because it doubles as the output layer.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n
