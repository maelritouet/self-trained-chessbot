// Tests for web/js/player.js, the JS port of chessgpt/player.py.
// Run: npm test   (from web/, with tools/node/bin on PATH)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Chess } from "chess.js";
import * as ort from "onnxruntime-node";

import { buildPrompt, chooseMove, formatMovetext, onnxRunner, Tokenizer } from "../js/player.js";

const ITOS = ["\n", ..." #+-./0123456789=BKNOQRabcdefghx"];
const TOK = new Tokenizer(ITOS);
const V = ITOS.length;

// --- helpers -------------------------------------------------------------------------------
/** GPT-interface runner with all logits zero: uniform next-character distribution. */
function uniformRunner(rows) {
  const B = rows.length, T = rows[0].length;
  return Promise.resolve({ data: new Float32Array(B * T * V), dims: [B, T, V] });
}

/** Oracle runner: puts all mass on the character following the input prefix in `text`. */
function oracleRunner(text, { maxT = Infinity } = {}) {
  return async (rows) => {
    const B = rows.length, T = rows[0].length;
    assert.ok(T <= maxT, `model got ${T} tokens, more than block size ${maxT}`);
    const data = new Float32Array(B * T * V).fill(-20);
    rows.forEach((row, b) => {
      const seen = TOK.decode(row);
      for (let t = 0; t < T; t++) {
        const prefix = seen.slice(0, t + 1);
        const next = text.startsWith(prefix) && prefix.length < text.length ? text[prefix.length] : "\n";
        data[(b * T + t) * V + ITOS.indexOf(next)] = 20;
      }
    });
    return { data, dims: [B, T, V] };
  };
}

/** Deterministic PRNG (mulberry32) for reproducible sampling. */
function seededRng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const base = { tokenizer: TOK, Chess, blockSize: 1024 };

// --- prompt format (same literals as the Python tests) -------------------------------------------
test("formatMovetext numbers white moves", () => {
  assert.equal(formatMovetext(["e4", "e5", "Nf3"]), "1.e4 e5 2.Nf3");
  assert.equal(formatMovetext([]), "");
});

test("buildPrompt ends where the next move starts", () => {
  assert.equal(buildPrompt([], "1-0"), "\n1-0 1.");
  assert.equal(buildPrompt(["e4"], "1-0"), "\n1-0 1.e4 ");
  assert.equal(buildPrompt(["e4", "e5"], "1-0"), "\n1-0 1.e4 e5 2.");
  assert.equal(buildPrompt(["e4"], null), "\n1.e4 ");
});

test("tokenizer round-trips and rejects unknown characters", () => {
  const s = "\n1/2-1/2 1.e4 e5 2.Nf3 Nc6 3.O-O e8=Q+ Qxf7#";
  assert.equal(TOK.decode(TOK.encode(s)), s);
  assert.throws(() => TOK.encode("Z"), /not in vocab/);
});

// --- chooseMove behaviour ---------------------------------------------------------------------
test("uniform model: candidate probabilities follow the character-product formula", async () => {
  const moves = ["e4", "e5"];
  const c = await chooseMove({ ...base, runModel: uniformRunner, moves, temperature: 0, maxRetries: 0 });
  // P(move) = (1/V)^len(san) * P(boundary) with 2 boundary chars (" " and "\n").
  const expected = (san) => (1 / V) ** san.length * (2 / V);
  assert.equal(c.candidates.length, 5);
  for (const { san, prob } of c.candidates) assert.ok(Math.abs(prob - expected(san)) < 1e-12, san);
  const board = new Chess(); moves.forEach((m) => board.move(m));
  const total = board.moves().reduce((s, san) => s + expected(san), 0);
  assert.ok(Math.abs(c.legalMass - total) < 1e-12);
  const probs = c.candidates.map((x) => x.prob);
  assert.deepEqual(probs, [...probs].sort((a, b) => b - a));
});

test("a legal sampled move is returned with no illegal tries", async () => {
  const runModel = oracleRunner("\n1-0 1.e4 e5 2.Nf3 ");
  const c = await chooseMove({ ...base, runModel, moves: ["e4", "e5"], resultPrefix: "1-0", temperature: 0 });
  assert.equal(c.san, "Nf3");
  assert.deepEqual(c.illegalTries, []);
  assert.equal(c.candidates[0].san, "Nf3");
  assert.ok(c.candidates[0].prob > 0.99);
});

test("illegal output is retried, then null with every try recorded", async () => {
  const runModel = oracleRunner("\n1-0 1.e4 e5 2.Nf6 ");
  const c = await chooseMove({ ...base, runModel, moves: ["e4", "e5"], resultPrefix: "1-0", temperature: 0, maxRetries: 2 });
  assert.equal(c.san, null);
  assert.deepEqual(c.illegalTries, ["Nf6", "Nf6", "Nf6"]);
});

test("sampling with the same seeded rng is reproducible", async () => {
  const args = { ...base, runModel: uniformRunner, moves: ["e4"], temperature: 1, maxRetries: 3 };
  const a = await chooseMove({ ...args, rng: seededRng(7) });
  const b = await chooseMove({ ...args, rng: seededRng(7) });
  assert.deepEqual(a.illegalTries, b.illegalTries);
  assert.equal(a.illegalTries.length + (a.san ? 1 : 0), a.san ? a.illegalTries.length + 1 : 4);
});

test("a finished game is rejected", async () => {
  const moves = ["f3", "e5", "g4", "Qh4#"];
  await assert.rejects(chooseMove({ ...base, runModel: uniformRunner, moves }), /game is already over/);
});

test("prompts longer than blockSize are cropped to the last blockSize tokens", async () => {
  const moves = ["Nf3", "Nf6", "Ng1", "Ng8"].flatMap((m) => [m]).concat(["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6"]);
  const runModel = oracleRunner("unused", { maxT: 24 });
  const c = await chooseMove({ ...base, blockSize: 24, runModel, moves, temperature: 0, maxRetries: 0 });
  assert.equal(c.candidates.length, 5);
});

// --- parity with the Python implementation ----------------------------------------------------
test("matches Python choose_move on every fixture case (tiny trained model)", async () => {
  const dir = new URL("./fixtures/", import.meta.url);
  const meta = JSON.parse(readFileSync(new URL("tiny/meta.json", dir)));
  const { max_retries: maxRetries, cases } = JSON.parse(readFileSync(new URL("cases.json", dir)));
  const session = await ort.InferenceSession.create(new URL("tiny/model.onnx", dir).pathname);
  const runModel = onnxRunner(ort, session);
  const tokenizer = new Tokenizer(meta.itos);

  for (const [i, py] of cases.entries()) {
    const js = await chooseMove({
      runModel, tokenizer, Chess, moves: py.moves, temperature: 0, maxRetries, topK: 10_000,
      resultPrefix: py.result_prefix, delimiter: meta.delimiter, blockSize: meta.block_size,
    });
    const where = `case ${i} (${py.moves.length} plies, prefix ${py.result_prefix})`;
    assert.equal(js.san, py.san, `${where}: chosen move`);
    assert.deepEqual(js.illegalTries, py.illegal_tries, `${where}: illegal tries`);
    const pyProbs = new Map(py.candidates);
    assert.deepEqual(new Set(js.candidates.map((c) => c.san)), new Set(pyProbs.keys()), `${where}: legal SAN set`);
    for (const { san, prob } of js.candidates) {
      const ref = pyProbs.get(san);
      assert.ok(Math.abs(prob - ref) <= 1e-4 * ref + 1e-7, `${where}: P(${san}) js ${prob} vs py ${ref}`);
    }
    assert.ok(Math.abs(js.legalMass - py.legal_mass) <= 1e-4 * py.legal_mass + 1e-7, `${where}: legal mass`);
  }
});
