// Game state behind the page: whose turn, legal targets, promotions, end of game.
// No DOM and no model here, so it is tested in Node (web/test/game.test.mjs).
// chess.js is passed in, like in player.js, so the same file runs in Node and the browser.

const RESULT_FOR_WINNER = { w: "1-0", b: "0-1" };

export class Game {
  /** humanColor: "w" or "b". moves: SAN moves already played (for tests or resuming). */
  constructor({ Chess, humanColor, moves = [] }) {
    this.board = new Chess();
    this.humanColor = humanColor;
    for (const san of moves) this.board.move(san);
  }

  get moves() { return this.board.history(); }
  get turn() { return this.board.turn(); }
  get isHumanTurn() { return !this.status.over && this.turn === this.humanColor; }

  /** The result the bot is prompted with: its own win (see decision 001). */
  get botResultPrefix() { return RESULT_FOR_WINNER[this.humanColor === "w" ? "b" : "w"]; }

  /** Squares the piece on `square` can move to, if it is the human's piece on the human's turn. */
  legalTargets(square) {
    if (!this.isHumanTurn) return [];
    const piece = this.board.get(square);
    if (!piece || piece.color !== this.humanColor) return [];
    return [...new Set(this.board.moves({ square, verbose: true }).map((m) => m.to))].sort();
  }

  /** Play the human's move. Returns {ok, san} or {ok: false, needsPromotion?}. */
  tryHumanMove(from, to, promotion) {
    if (!this.isHumanTurn) return { ok: false };
    const options = this.board.moves({ square: from, verbose: true }).filter((m) => m.to === to);
    if (!options.length) return { ok: false };
    if (options.some((m) => m.promotion) && !promotion) return { ok: false, needsPromotion: true };
    const move = options.find((m) => !m.promotion || m.promotion === promotion);
    if (!move) return { ok: false };
    this.board.move(move.san);
    return { ok: true, san: move.san };
  }

  /** Play the bot's move; it must be the exact SAN of a legal move. */
  applyBotMove(san) {
    if (!this.board.moves().includes(san)) throw new Error(`not a legal move here: ${san}`);
    this.board.move(san);
  }

  /** End-of-game state. Unlike the move function, the game loop ends claimable draws
   *  (threefold repetition, 50-move rule) automatically, as Lichess does. */
  get status() {
    const b = this.board;
    if (b.isCheckmate()) return { over: true, result: RESULT_FOR_WINNER[b.turn() === "w" ? "b" : "w"], reason: "checkmate" };
    const draw = (reason) => ({ over: true, result: "1/2-1/2", reason });
    if (b.isStalemate()) return draw("stalemate");
    if (b.isInsufficientMaterial()) return draw("insufficient material");
    if (b.isThreefoldRepetition()) return draw("threefold repetition");
    if (b.isDrawByFiftyMoves()) return draw("fifty-move rule");
    return { over: false, result: null, reason: null };
  }
}

/** The move the bot actually plays: its own legal move, or, if every try was illegal,
 *  its highest-scored legal move (flagged, so the page can say so). */
export function resolveBotMove(choice) {
  if (choice.san) return { san: choice.san, fallback: false };
  return { san: choice.candidates[0].san, fallback: true };
}
