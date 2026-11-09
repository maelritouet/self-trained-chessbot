"""Lichess puzzle accuracy (W4, not implemented yet).

Plan: load the Lichess puzzle CSV (https://database.lichess.org/#puzzles).
For each puzzle, replay its position as movetext and have the model play the
solution moves. Report accuracy bucketed by puzzle rating.

Caveat: puzzles start from a FEN, not from move 1, while the model only sees
movetext. Use the puzzle's source game (GameUrl) to rebuild the move history.
"""
