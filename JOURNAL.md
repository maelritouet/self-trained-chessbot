# Journal

Weekly entries, 5 lines each: what I did, what I learned, what broke, what I decided, what's next.

## Week 0 — 2026-11-09

- **Did:** set up the repo, a streaming Lichess pipeline, char tokenizer, GPT model + training loop, eval metrics and the Kaggle workflow (all test-first, mutation-checked); first 30-min run on a T4; a working browser demo whose JS move function matches the Python one on 14 test positions.
- **Learned:** Kaggle retired the P100, so runs use a T4 at ~100k tokens/s (~4B tokens per 11 h session). The 8M-token sample overfit after ~18 passes (best val loss 0.574), yet that model already opens like a human (Italian Game; 63% Nc6 after 1.e4 e5 2.Nf3) and plays a legal first move in 81% of held-out positions, falling from 96% in the opening to 63% by move 41.
- **Broke:** python-chess and chess.js both accept non-SAN text (`Nxf3` for `Nf3`, `--` as a null move!), so the bot could have played a pass: legality is now exact SAN everywhere. Also: the Flatpak sandbox hid tools, a PC crash killed the 3M download at 60% (long jobs now run detached), and GitHub's first commit leaked my email (history rewritten).
- **Decided:** result token first, newline delimiter, Elo ≥ 1800, blitz/rapid/classical, drop unfinished games, block_size 1024, 3M games (decisions 001–007); at 1800+ the data is 82% blitz and 99.5% of games fit the context.
- **Next:** build the 3M-game dataset (~1.26B tokens), first real run (`base.yaml`, ~1.8 h), measure it with the metrics, then the W3 controlled experiment.
