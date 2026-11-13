// Page wiring: board, controls, model loading, and the bot's turn.
// The logic lives in tested modules (player.js: the move function, game.js: game state);
// this file only connects them to the DOM.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.mjs";
import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.4.0/dist/esm/chess.js";
import { buildPrompt, chooseMove, onnxRunner, Tokenizer } from "./player.js";
import { Game, resolveBotMove } from "./game.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
// Multi-threaded WASM needs a cross-origin-isolated page (COOP/COEP headers), which static
// hosts like GitHub Pages can't set; fall back to one thread there.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const MAX_RETRIES = 5;
const CONTEXT_CHARS = 72; // how much of the prompt the pen strip shows
const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const NAME = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const FILES = "abcdefgh";

const $ = (id) => document.getElementById(id);
const el = {
  board: $("board"), status: $("status"), progress: $("load-progress"), promotion: $("promotion"),
  newGame: $("new-game"), colorButtons: document.querySelectorAll(".segmented button"),
  modelField: $("model-field"), modelSelect: $("model-select"),
  temperature: $("temperature"), tempValue: $("temp-value"),
  pen: document.querySelector(".pen"), penContext: $("pen-context"), penTries: $("pen-tries"),
  penWriting: $("pen-writing"), penNote: $("pen-note"),
  candidates: $("candidates"), legalMass: $("legal-mass"), moves: $("moves"), footnote: $("footnote"),
};

const state = {
  model: null,            // {runModel, tokenizer, meta}
  game: null,
  humanColor: "w",
  selected: null,
  lastMove: null,         // {from, to}
  fallbackPlies: new Set(),
  thinking: false,
  gameId: 0,              // bumps on new game, so a stale bot reply is dropped
  models: [],             // the menu: [{id, label, base, files}]
  loadId: 0,              // bumps on each model load, so a slower earlier download is dropped
};

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

// --- model loading ---------------------------------------------------------------------------
async function fetchWithProgress(url, onProgress, size) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = size || Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

// Models live on the Hugging Face Hub (decision 014); models.release.json lists them with their
// SHA-256. A local copy from scripts/fetch_models.py is the fallback, or forced with ?local.
const MANIFEST = "models.release.json";
const FORCE_LOCAL = new URLSearchParams(location.search).has("local");
const CACHE_NAME = "chessbot-models-v1";
const hfBase = (repo, id) => `https://huggingface.co/${repo}/resolve/main/${id}/`;
const localBase = (id) => `models/${id}/`;

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Throw unless bytes match the manifest entry (skipped for models not in the manifest, and
 *  where the page isn't a secure context, since crypto.subtle needs https or localhost). */
async function verify(bytes, info, name) {
  if (!info) return;
  if (bytes.length !== info.size) throw new Error(`${name}: got ${bytes.length} bytes, expected ${info.size}`);
  if (crypto.subtle && (await sha256Hex(bytes)) !== info.sha256) throw new Error(`${name}: checksum mismatch`);
}

/** Bytes of url, from the browser cache if a checked copy is there, else downloaded,
 *  checked and cached. The cache key holds the checksum, so a new model version is a new key. */
async function fetchChecked(url, info, onProgress) {
  const key = info ? `${url}?sha256=${info.sha256}` : null;
  const cache = key && self.caches ? await caches.open(CACHE_NAME).catch(() => null) : null;
  const hit = cache && (await cache.match(key));
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    try { await verify(bytes, info, url); return { bytes, cached: true }; } catch { await cache.delete(key); }
  }
  const bytes = await fetchWithProgress(url, onProgress, info?.size);
  await verify(bytes, info, url);
  if (cache) await cache.put(key, new Response(bytes)).catch(() => {}); // full disk: just don't cache
  return { bytes, cached: false };
}

function showProgress(fraction) {
  el.progress.hidden = fraction === null;
  if (fraction !== null) el.progress.value = fraction;
}

async function loadModel(entry) {
  const files = entry.files || {};
  const meta = JSON.parse(new TextDecoder().decode(
    (await fetchChecked(entry.base + "meta.json", files["meta.json"], () => {})).bytes));
  const mb = (n) => (n / 1e6).toFixed(0);
  showProgress(0);
  const { bytes, cached } = await fetchChecked(entry.base + "model.onnx", files["model.onnx"], (got, total) => {
    showProgress(total ? got / total : null);
    el.status.textContent = total
      ? `Downloading the model (once; it is kept in your browser): ${mb(got)} of ${mb(total)} MB`
      : `Downloading the model: ${mb(got)} MB`;
  });
  showProgress(null);
  el.status.textContent = cached ? "Starting the model (from your browser's cache)…" : "Starting the model…";
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const run = onnxRunner(ort, session);
  // Yield a frame before each forward pass so the pen strip can repaint between characters.
  const runModel = async (rows) => { await nextFrame(); return run(rows); };
  return { runModel, tokenizer: new Tokenizer(meta.itos), meta };
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

/** The model menu: [{id, label, base, files}] where base is the folder URL and files the
 *  checksums (absent for a hand-added local model). */
async function loadModelList() {
  const manifest = await fetchJson(MANIFEST);
  if (FORCE_LOCAL) return localModelList(manifest);
  return manifest.models.map((m) => ({ ...m, base: hfBase(manifest.hf_repo, m.id) }));
}

/** Models from web/models/models.json (written by scripts/fetch_models.py, or by hand). */
async function localModelList(manifest) {
  const byId = new Map(manifest.models.map((m) => [m.id, m]));
  return (await fetchJson("models/models.json"))
    .map((m) => ({ ...m, base: localBase(m.id), files: byId.get(m.id)?.files }));
}

/** Load entry; if the Hub can't be reached, try the same model from a local copy. */
async function loadModelWithFallback(entry) {
  try {
    return await loadModel(entry);
  } catch (err) {
    if (entry.base === localBase(entry.id)) throw err;
    console.warn(`Hugging Face download failed (${err.message}); trying the local copy.`);
    return loadModel({ ...entry, base: localBase(entry.id) }).catch(() => { throw err; });
  }
}

// --- board -----------------------------------------------------------------------------------
function squaresInView() {
  const ranks = state.humanColor === "w" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = state.humanColor === "w" ? [...FILES] : [...FILES].reverse();
  return ranks.flatMap((r) => files.map((f) => `${f}${r}`));
}

function renderBoard() {
  const { game, selected, lastMove } = state;
  const targets = selected ? game.legalTargets(selected) : [];
  const view = squaresInView();
  el.board.replaceChildren(...view.map((sq, i) => {
    const file = FILES.indexOf(sq[0]), rank = Number(sq[1]);
    const piece = game.board.get(sq);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "square" + ((file + rank) % 2 === 0 ? " dark" : "");
    b.dataset.square = sq;
    if (lastMove && (sq === lastMove.from || sq === lastMove.to)) b.classList.add("last");
    if (sq === selected) b.classList.add("selected");
    if (targets.includes(sq)) b.classList.add("target");
    if (piece) {
      b.classList.add("occupied");
      if (game.legalTargets(sq).length) b.classList.add("movable");
      const p = document.createElement("span");
      p.className = `piece ${piece.color}`;
      p.textContent = GLYPH[piece.type];
      p.setAttribute("aria-hidden", "true");
      b.append(p);
    }
    b.setAttribute("aria-label", `${sq}, ${piece ? `${piece.color === "w" ? "white" : "black"} ${NAME[piece.type]}` : "empty"}`
      + (targets.includes(sq) ? ", legal target" : ""));
    if (i % 8 === 0) b.append(coord("rank", sq[1]));
    if (i >= 56) b.append(coord("file", sq[0]));
    return b;
  }));
}

function coord(kind, text) {
  const s = document.createElement("span");
  s.className = `coord ${kind}`;
  s.textContent = text;
  s.setAttribute("aria-hidden", "true");
  return s;
}

el.board.addEventListener("click", (event) => {
  const sq = event.target.closest(".square")?.dataset.square;
  if (!sq || state.thinking || !state.game || state.game.status.over) return;
  const { game, selected } = state;
  if (selected && game.legalTargets(selected).includes(sq)) {
    humanMove(selected, sq);
  } else {
    state.selected = game.legalTargets(sq).length ? sq : null;
    renderBoard();
    if (state.selected) el.board.querySelector(`[data-square="${sq}"]`)?.focus();
  }
});

function askPromotion(color) {
  return new Promise((resolve) => {
    const box = el.promotion.querySelector(".promotion-choices");
    box.replaceChildren(...["q", "r", "b", "n"].map((t) => {
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = `<span class="piece ${color}">${GLYPH[t]}</span>`;
      b.setAttribute("aria-label", NAME[t]);
      b.addEventListener("click", () => { el.promotion.hidden = true; resolve(t); });
      return b;
    }));
    el.promotion.hidden = false;
    box.firstChild.focus();
  });
}

async function humanMove(from, to) {
  const { game } = state;
  let res = game.tryHumanMove(from, to);
  if (res.needsPromotion) res = game.tryHumanMove(from, to, await askPromotion(state.humanColor));
  state.selected = null;
  if (!res.ok) { renderBoard(); return; }
  state.lastMove = { from, to };
  renderAll();
  await botTurn();
}

// --- the bot's turn ---------------------------------------------------------------------------
function displayPrompt(moves, prefix) {
  const text = buildPrompt(moves, prefix, state.model.meta.delimiter).replace(/^\n/, "");
  return text.length > CONTEXT_CHARS ? "…" + text.slice(-CONTEXT_CHARS) : text;
}

const TOP_N = 5;
const ordinal = (n) => n + ({ 1: "st", 2: "nd", 3: "rd" }[n % 100 > 10 && n % 100 < 14 ? 0 : n % 10] || "th");

/** Top five candidates, plus the played move as an extra row if it ranked lower,
 *  so the chart never hides what the bot actually did. */
function renderCandidates(all, playedSan) {
  const rows = all.slice(0, TOP_N).map((c) => ({ ...c, rank: null }));
  const playedRank = all.findIndex((c) => c.san === playedSan);
  if (playedRank >= TOP_N) rows.push({ ...all[playedRank], rank: playedRank + 1 });
  el.candidates.replaceChildren(...rows.map(({ san, prob, rank }) => {
    const li = document.createElement("li");
    li.className = "bar" + (san === playedSan ? " played" : "") + (rank ? " outside" : "");
    const pct = 100 * prob;
    li.innerHTML = `<span class="bar-san"></span><span class="bar-track"><span class="bar-fill"></span></span><span class="bar-pct"></span>`;
    li.querySelector(".bar-san").textContent = san;
    li.querySelector(".bar-fill").style.width = `${Math.max(pct, 0.5)}%`;
    li.querySelector(".bar-pct").textContent = pct >= 1 ? `${pct.toFixed(0)}%` : pct >= 0.1 ? `${pct.toFixed(1)}%` : "<0.1%";
    if (rank) {
      const note = document.createElement("span");
      note.className = "bar-rank";
      note.textContent = `played: its ${ordinal(rank)} choice`;
      li.append(note);
    }
    li.setAttribute("aria-label", `${san}: ${pct.toFixed(1)} percent${san === playedSan ? `, played${rank ? `, its ${ordinal(rank)} choice` : ""}` : ""}`);
    return li;
  }));
}

async function botTurn() {
  const { game, model } = state;
  if (game.status.over || game.isHumanTurn) return finishIfOver();
  const id = state.gameId;
  state.thinking = true;
  el.status.textContent = "Thinking…";
  el.pen.classList.add("writing");
  el.penContext.textContent = displayPrompt(game.moves, game.botResultPrefix);
  el.penTries.replaceChildren();
  el.penWriting.textContent = "";
  el.penWriting.classList.remove("played");

  const choice = await chooseMove({
    runModel: model.runModel, tokenizer: model.tokenizer, Chess, moves: game.moves,
    temperature: Number(el.temperature.value), maxRetries: MAX_RETRIES, topK: Infinity,
    resultPrefix: game.botResultPrefix, delimiter: model.meta.delimiter, blockSize: model.meta.block_size,
    onEvent: (e) => {
      if (id !== state.gameId) return;
      if (e.type === "scored") {
        renderCandidates(e.candidates, null);
        el.legalMass.textContent = `${(100 * e.legalMass).toFixed(0)}% of its probability went to legal moves.`;
      } else if (e.type === "char") {
        el.penWriting.textContent = e.text;
      } else if (e.type === "try" && !e.legal) {
        const s = document.createElement("s");
        s.textContent = e.text || "(nothing)";
        s.title = "Illegal: not a legal move in this position";
        el.penTries.append(s);
        el.penWriting.textContent = "";
      }
    },
  });
  if (id !== state.gameId) return; // a new game started meanwhile

  const { san, fallback } = resolveBotMove(choice);
  game.applyBotMove(san);
  const last = game.board.history({ verbose: true }).at(-1);
  state.lastMove = { from: last.from, to: last.to };
  if (fallback) state.fallbackPlies.add(game.moves.length - 1);
  el.penWriting.textContent = san;
  el.penWriting.classList.add("played");
  el.pen.classList.remove("writing");
  el.penNote.textContent = fallback
    ? `All ${MAX_RETRIES + 1} attempts were illegal, so it played its highest-scored legal move.`
    : choice.illegalTries.length
      ? `It wrote ${choice.illegalTries.length} illegal move${choice.illegalTries.length > 1 ? "s" : ""} before a legal one.`
      : "Its first attempt was legal.";
  renderCandidates(choice.candidates, san);
  state.thinking = false;
  renderAll();
  finishIfOver();
}

function finishIfOver() {
  const st = state.game.status;
  if (!st.over) {
    el.status.textContent = state.game.isHumanTurn ? "Your move." : "Thinking…";
    return;
  }
  const humanWon = st.result === (state.humanColor === "w" ? "1-0" : "0-1");
  el.status.textContent = st.result === "1/2-1/2"
    ? `Draw by ${st.reason}.`
    : `${st.reason === "checkmate" ? "Checkmate" : "Game over"}. ${humanWon ? "You win." : "The bot wins."}`;
}

// --- scoresheet ------------------------------------------------------------------------------
function renderScoresheet() {
  const moves = state.game.moves;
  const rows = [];
  for (let i = 0; i < moves.length; i += 2) {
    const tr = document.createElement("tr");
    const num = document.createElement("td");
    num.textContent = `${i / 2 + 1}.`;
    tr.append(num, cell(i, moves[i]), cell(i + 1, moves[i + 1]));
    rows.push(tr);
  }
  el.moves.replaceChildren(...rows);
  el.moves.scrollTop = el.moves.scrollHeight;
  el.footnote.hidden = state.fallbackPlies.size === 0;
}

function cell(ply, san) {
  const td = document.createElement("td");
  if (san === undefined) return td;
  const byBot = (ply % 2 === 0 ? "w" : "b") !== state.humanColor;
  td.className = byBot ? "bot" : "human";
  td.textContent = san + (state.fallbackPlies.has(ply) ? "*" : "");
  return td;
}

function renderAll() { renderBoard(); renderScoresheet(); }

// --- controls --------------------------------------------------------------------------------
async function newGame() {
  state.gameId += 1;
  state.game = new Game({ Chess, humanColor: state.humanColor });
  Object.assign(state, { selected: null, lastMove: null, fallbackPlies: new Set(), thinking: false });
  el.pen.classList.remove("writing");
  el.penContext.textContent = "";
  el.penTries.replaceChildren();
  el.penWriting.textContent = "";
  el.penNote.textContent = "Its moves appear here as it types them. Illegal attempts are crossed out and it tries again.";
  el.candidates.replaceChildren();
  el.legalMass.textContent = "";
  renderAll();
  if (state.game.isHumanTurn) el.status.textContent = "Your move.";
  else await botTurn();
}

el.colorButtons.forEach((button) => button.addEventListener("click", () => {
  el.colorButtons.forEach((b) => b.setAttribute("aria-checked", String(b === button)));
  state.humanColor = button.dataset.color;
  if (state.model) newGame();
}));
el.newGame.addEventListener("click", () => state.model && newGame());
el.temperature.addEventListener("input", () => { el.tempValue.textContent = Number(el.temperature.value).toFixed(1); });
el.modelSelect.addEventListener("change", async () => {
  state.gameId += 1;
  state.model = null;
  const load = ++state.loadId;
  try {
    const model = await loadModelWithFallback(state.models.find((m) => m.id === el.modelSelect.value));
    if (load !== state.loadId) return; // another model was picked while this one downloaded
    state.model = model;
    await newGame();
  } catch (err) { if (load === state.loadId) showError(err); }
});

function showError(err) {
  console.error(err);
  showProgress(null);
  el.status.textContent = `The model could not be loaded (${err.message}). Check your connection and `
    + "reload, or play offline: python3 scripts/fetch_models.py, then open the page with ?local (see the README).";
}

// --- start ---------------------------------------------------------------------------------------
(async function start() {
  state.game = new Game({ Chess, humanColor: state.humanColor });
  renderAll();
  try {
    state.models = await loadModelList();
    el.modelSelect.replaceChildren(...state.models.map((m) => new Option(m.label, m.id)));
    el.modelField.hidden = state.models.length < 2;
    const load = ++state.loadId;
    const model = await loadModelWithFallback(state.models[0]);
    if (load !== state.loadId) return;
    state.model = model;
    await newGame();
  } catch (err) { showError(err); }
})();
