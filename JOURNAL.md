# Journal

Weekly entries, 5 lines each: what I did, what I learned, what broke, what I decided, what's next.

## Week 0 — 2026-11-09

- **Did:** set up the repo, a streaming Lichess pipeline, char tokenizer, GPT model + training loop, eval metrics and the Kaggle workflow (all test-first, mutation-checked); first 30-min run on a T4; a working browser demo whose JS move function matches the Python one on 14 test positions.
- **Learned:** Kaggle retired the P100, so runs use a T4 at ~100k tokens/s (~4B tokens per 11 h session). The 8M-token sample overfit after ~18 passes (best val loss 0.574), yet that model already opens like a human (Italian Game; 63% Nc6 after 1.e4 e5 2.Nf3) and plays a legal first move in 81% of held-out positions, falling from 96% in the opening to 63% by move 41.
- **Broke:** python-chess and chess.js both accept non-SAN text (`Nxf3` for `Nf3`, `--` as a null move!), so the bot could have played a pass: legality is now exact SAN everywhere. Also: the Flatpak sandbox hid tools, a PC crash killed the 3M download at 60% (long jobs now run detached), and GitHub's first commit leaked my email (history rewritten).
- **Decided:** result token first, newline delimiter, Elo ≥ 1800, blitz/rapid/classical, drop unfinished games, block_size 1024, 3M games (decisions 001–007); at 1800+ the data is 82% blitz and 99.5% of games fit the context.
- **Next:** build the 3M-game dataset (~1.26B tokens), first real run (`base.yaml`, ~1.8 h), measure it with the metrics, then the W3 controlled experiment.

## Week 1 — 2026-11-12

- **Did:** built the 3M-game dataset (1.26B tokens) and trained two models on Kaggle T4s: base (10.7M params, 3.93B tokens, 8.6 h) and large (25.2M, 1.97B tokens, 9.1 h). Wrote the eval harness (strict legal rate by move number, 1,000 Lichess puzzles by rating, 150 games vs Stockfish 19 Skill 0/1/2) and fine-tuned both models on 594k long games from another month. ~21 of 30 weekly GPU hours used.
- **Learned:** at equal compute the bigger model wins everywhere despite seeing half the tokens (illegal outputs −47%, puzzles 8.0 → 11.3%); neither run overfit, so model size, not data, was the limit. Fine-tuning is a trade-off, not a free win: both models forgot a little (val +0.008) and lost puzzles, and only the large one gained where it was aimed (forfeits 39 → 28).
- **Broke:** local evals died when the VS Code session ended, even detached; the first fine-tune run failed its own smoke test, which checked that loss fell (a fine-tune starts near its minimum, so now it checks the weights loaded); python-chess's `claim_draw` ends a game one ply before the threefold repetition actually happens; the puzzle CSV uses a different game-URL format than expected.
- **Decided:** model size as the experiment, at equal compute (009); "Stockfish level 1–3" = Skill Level 0/1/2 at 50k nodes, with forfeits counted separately (011); fine-tune on long games for late-game legality (012), kept as a mixed result; ship both base and large in the web demo.
- **Next:** decision outcomes for 009 and 012; export the three new models to the web demo; temperature slider and top-5 check; choose hosting (large's ONNX is ~100 MB, GitHub Pages' file limit).
