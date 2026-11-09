"""Character-level tokenizer: one token per character, vocab built from the data.

The whole vocabulary is ~35 symbols (SAN letters, digits, '+#=-/.', space, the
game delimiter), so ids fit in uint8 and train.bin is 1 byte per character.

    tok = CharTokenizer.from_text(all_text)
    ids = tok.encode("1.e4 e5")      # list[int]
    tok.decode(ids)                  # "1.e4 e5"
    tok.save("data/processed/meta.json"); CharTokenizer.load(...)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocab")
        if any(len(c) != 1 for c in chars):
            raise ValueError("every vocab entry must be a single character")
        self.itos: list[str] = list(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def from_text(cls, text: str | set[str], extra: str = "") -> "CharTokenizer":
        """Vocab = sorted set of characters in `text` (plus `extra`, e.g. the delimiter).

        Sorting makes the id assignment deterministic for a given character set.
        """
        return cls(sorted(set(text) | set(extra)))

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} not in vocab") from None

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def encode_array(self, text: str) -> np.ndarray:
        """Fast bulk encode to a uint8 array (for building train.bin).

        Uses a 256-entry lookup table over the ASCII bytes instead of a Python
        loop over characters: ~100x faster on hundreds of MB of text.
        """
        if self.vocab_size > 256:
            raise ValueError("vocab too large for uint8")
        raw = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        lut = np.full(256, 255, dtype=np.uint8)  # 255 = "unknown" sentinel
        for c, i in self.stoi.items():
            lut[ord(c)] = i
        ids = lut[raw]
        if self.vocab_size <= 255 and (ids == 255).any():
            bad = sorted({chr(b) for b in raw[ids == 255]})
            raise ValueError(f"characters not in vocab: {bad}")
        return ids

    # --- persistence (meta.json) ---
    def to_meta(self) -> dict:
        return {"vocab_size": self.vocab_size, "itos": self.itos, "stoi": self.stoi}

    def save(self, path: str | Path, **extra) -> None:
        """Write meta.json: vocab plus any extra fields (counts, delimiter, ...)."""
        meta = {**self.to_meta(), **extra}
        Path(path).write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        meta = json.loads(Path(path).read_text())
        return cls(meta["itos"])
