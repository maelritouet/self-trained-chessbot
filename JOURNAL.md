# Journal

Weekly entries, 5 lines each: what I did, what I learned, what broke, what I decided, what's next.

## Week 0 — 2026-11-09

- **Did:** set up the repo, a streaming Lichess pipeline (20k-game sample, 1000/1000 replay legal), char tokenizer, GPT model + training loop (test-first, mutation-checked), and the Kaggle workflow; first 30-min run on a T4.
- **Learned:** Kaggle retired the P100, so runs use a T4 at ~100k tokens/s (~4B tokens per 11 h session). The 8M-token sample overfit after ~18 passes (best val loss 0.574), yet that model already opens like a human (Italian Game; 63% Nc6 after 1.e4 e5 2.Nf3) with 25% illegal greedy moves in self-play.
- **Broke:** the VS Code Flatpak sandbox hides `~/.local/bin` and the host Python; the Kaggle CLI forgot to declare `six`; a weak test that mutation testing exposed; GitHub's initial commit leaked my email, so history was rewritten.
- **Decided:** result token first, newline delimiter, Elo ≥ 1800, blitz/rapid/classical, drop unfinished games, block_size 1024, 3M games (decisions 001–007).
- **Next:** build the 3M-game dataset, first real run (`base.yaml`, ~1.8 h), eval metrics, then the W3 controlled experiment.

