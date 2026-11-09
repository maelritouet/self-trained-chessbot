"""Play the model against Stockfish (W4, not implemented yet).

Plan: use python-chess's UCI engine wrapper to play ~50 games per Stockfish
level (1-3), alternating colours, with moves chosen by
`chessgpt.player.choose_move`. Report score (W=1, D=0.5, L=0) and the number
of illegal attempts per game.
"""
