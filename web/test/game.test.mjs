// Tests for web/js/game.js: the game state behind the page (no DOM, no model).
import { test } from "node:test";
import assert from "node:assert/strict";
import { Chess } from "chess.js";

import { Game, resolveBotMove } from "../js/game.js";

const PROMO_LINE = ["e4", "d5", "exd5", "c6", "dxc6", "Nf6", "cxb7", "Nbd7"]; // White's b7 pawn can promote

test("legal targets of the human's piece, sorted", () => {
  const g = new Game({ Chess, humanColor: "w" });
  assert.deepEqual(g.legalTargets("e2"), ["e3", "e4"]);
  assert.deepEqual(g.legalTargets("g1"), ["f3", "h3"]);
});

test("no targets for the bot's pieces or an empty square", () => {
  const g = new Game({ Chess, humanColor: "w" });
  assert.deepEqual(g.legalTargets("e7"), []);
  assert.deepEqual(g.legalTargets("e4"), []);
});

test("a legal human move is played and recorded in SAN", () => {
  const g = new Game({ Chess, humanColor: "w" });
  assert.deepEqual(g.tryHumanMove("g1", "f3"), { ok: true, san: "Nf3" });
  assert.deepEqual(g.moves, ["Nf3"]);
  assert.equal(g.turn, "b");
});

test("an illegal human move is refused and changes nothing", () => {
  const g = new Game({ Chess, humanColor: "w" });
  assert.equal(g.tryHumanMove("e2", "e5").ok, false);
  assert.deepEqual(g.moves, []);
});

test("the human cannot move on the bot's turn", () => {
  const g = new Game({ Chess, humanColor: "b" });
  assert.equal(g.isHumanTurn, false);
  assert.equal(g.tryHumanMove("e2", "e4").ok, false);
  assert.deepEqual(g.legalTargets("e2"), []);
});

test("a promotion asks for the piece first, then plays it", () => {
  const g = new Game({ Chess, humanColor: "w", moves: PROMO_LINE });
  assert.deepEqual(g.tryHumanMove("b7", "a8"), { ok: false, needsPromotion: true });
  assert.deepEqual(g.moves, PROMO_LINE);
  assert.deepEqual(g.tryHumanMove("b7", "a8", "n"), { ok: true, san: "bxa8=N" });
});

test("bot moves are applied by SAN", () => {
  const g = new Game({ Chess, humanColor: "b" });
  g.applyBotMove("d4");
  assert.deepEqual(g.moves, ["d4"]);
  assert.equal(g.isHumanTurn, true);
  assert.throws(() => g.applyBotMove("d5d6"));
});

test("the bot is conditioned on its own win", () => {
  assert.equal(new Game({ Chess, humanColor: "w" }).botResultPrefix, "0-1");
  assert.equal(new Game({ Chess, humanColor: "b" }).botResultPrefix, "1-0");
});

test("status reports checkmate with the winner", () => {
  const g = new Game({ Chess, humanColor: "w", moves: ["f3", "e5", "g4", "Qh4#"] });
  assert.deepEqual(g.status, { over: true, result: "0-1", reason: "checkmate" });
});

test("status ends the game on threefold repetition", () => {
  const g = new Game({ Chess, humanColor: "w", moves: ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"] });
  assert.deepEqual(g.status, { over: true, result: "1/2-1/2", reason: "threefold repetition" });
});

test("status of a game in progress", () => {
  assert.deepEqual(new Game({ Chess, humanColor: "w", moves: ["e4"] }).status, { over: false, result: null, reason: null });
});

test("resolveBotMove uses the model's move, or falls back to its top legal candidate", () => {
  const cands = [{ san: "Nf3", prob: 0.4 }, { san: "d4", prob: 0.3 }];
  assert.deepEqual(resolveBotMove({ san: "d4", candidates: cands }), { san: "d4", fallback: false });
  assert.deepEqual(resolveBotMove({ san: null, candidates: cands }), { san: "Nf3", fallback: true });
});
