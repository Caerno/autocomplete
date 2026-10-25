// Движок автопродолжения в браузере: Web Worker, onnxruntime-web (WASM), модель v2.1 (P6) в шести графах ONNX.
// Построчный перенос web/py/runner.py (эталон; паритет — web/tests/test_site.py). Состояние модели живёт здесь, страница шлёт текст поля.
// Сообщения: {cmd:'init', base, threads} → {ev:'progress'|'ready'|'error'}; {cmd:'complete', id, text} → {ev:'answer', id, ...};
// {cmd:'trace', id, text} → лог-вероятности каждой следующей буквы и последней строки целиком (для тестов паритета).
importScripts('ort/ort.wasm.min.js');

const BEAM = 4, MAXLEN = 12, TOP = 5, CTX = 256, RESET_AT = 4096, KEEP_WORDS = 8;
const SCALES = [4, 6, 8, 12, 16, 24], NF = 3;
let G = {}, mt = null, H, DH, V, L, CHUNK, START, BODY0, Z0, ch2id, alpha, spacelike, vowel, sess = null;

const T = (type, data, dims) => new ort.Tensor(type, data, dims);
const i64 = v => T('int64', BigInt64Array.from([BigInt(v)]), [1]);

async function fetchBuf(url, onBytes) {                       // загрузка с прогрессом
  const r = await fetch(url); if (!r.ok) throw new Error(url + ': ' + r.status);
  const rd = r.body.getReader(), parts = []; let n = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); n += value.length; onBytes(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out;
}

async function init(base, threads, bodyFile) {
  ort.env.logLevel = 'error';                                  // предупреждения рантайма (например, «Unknown CPU vendor») — не в консоль
  ort.env.wasm.wasmPaths = new URL('ort/', self.location).href;    // абсолютный адрес: относительный ort грузит как имя модуля и падает
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, threads | 0) : 1;   // потоки — только при изоляции (COOP/COEP); по умолчанию 1: графы мелкие, 1 поток быстрее
  base = new URL(base, self.location).href;
  mt = await (await fetch(base + 'meta.json')).json();
  H = mt.heads; DH = mt.dh; V = mt.vocab; L = mt.local; CHUNK = mt.body.chunk;
  START = T('float32', Float32Array.from(mt.start), [1, 1, mt.C]);
  BODY0 = mt.body.state.map(nm => { const s = mt.body.shapes[nm]; return T('float32', new Float32Array(s.reduce((a, b) => a * b, 1)), s); });
  Z0 = T('float32', new Float32Array(0), [1, H, 0, DH]);
  ch2id = new Map(mt.chars.map((c, i) => [c, mt.char0 + i])); alpha = mt.alpha; spacelike = mt.spacelike; vowel = mt.vowel;
  const files = { enc: 'enc.onnx', dec: 'dec.onnx', sread: 'sread.onnx', cond: 'cond.onnx', copy: 'copy.onnx', body: bodyFile || 'body.int8.onnx' };   // тело в int8 (43 МБ); fp32 (158 МБ) — для проверки паритета
  const sizes = {}; let total = 0, got = 0;
  await Promise.all(Object.entries(files).map(async ([k, f]) => { const h = await fetch(base + f, { method: 'HEAD' }); sizes[k] = +h.headers.get('content-length') || 0; total += sizes[k]; }));
  const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  for (const [k, f] of Object.entries(files)) {
    const buf = await fetchBuf(base + f, n => { got += n; postMessage({ ev: 'progress', got, total }); });
    G[k] = await ort.InferenceSession.create(buf, opt);
  }
  sess = new Session();
  postMessage({ ev: 'ready', threads: ort.env.wasm.numThreads, mb: Math.round(total / 1e6) });
}

async function run(nm, args) {
  const feeds = {}; mt.graphs[nm].in.forEach((k, i) => { feeds[k] = args[i]; });
  const out = await G[nm].run(feeds); return mt.graphs[nm].out.map(k => out[k]);
}

function rowsOf(t, a, b) {                                     // кэш (1,H,n,DH) → строки [a, b) (копия)
  const n = t.dims[2], m = b - a, out = new Float32Array(H * m * DH);
  for (let h = 0; h < H; h++) out.set(t.data.subarray((h * n + a) * DH, (h * n + b) * DH), h * m * DH);
  return T('float32', out, [1, H, m, DH]);
}
function cat1(list) {                                          // (1,1,C) × L → (1,L,C)
  const C = mt.C, out = new Float32Array(list.length * C); list.forEach((t, i) => out.set(t.data, i * C)); return T('float32', out, [1, list.length, C]);
}
function logSoftmax(x) {
  let m = -Infinity; for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
  let s = 0; for (let i = 0; i < x.length; i++) s += Math.exp(x[i] - m);
  const ls = m + Math.log(s), out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = x[i] - ls; return out;
}

class Runner {                                                // пошаговый вывод: буква → логиты следующей буквы (без головы E)
  constructor() { this.reset(); }
  reset() {
    this.pos = 0; this.cstart = 0; this.enc = [Z0, Z0, Z0, Z0]; this.dec = [Z0, Z0, Z0, Z0];
    this.prev_sp = true; this.lastb = -1; this.word_e = []; this.frag_cnt = 0; this.hint_mark = -1; this.prt = null;
    this.n_words = 0; this.body = BODY0.slice(); this.hb = null; this.cond = START; this.zn = null;
  }
  snapshot() { return { ...this, word_e: this.word_e.slice() }; }
  restore(d) { Object.assign(this, d); this.word_e = d.word_e.slice(); }
  async _cond() {
    if (this.n_words === 0) return START;
    if (mt.frag_refresh && this.hint_mark >= 0) {
      if (this.prt === null) {
        const E = cat1(this.word_e.slice(0, this.hint_mark + 1)), n = E.dims[1];
        this.prt = (await run('sread', [E, T('float32', Float32Array.from({ length: n }, (_, i) => i + 1), [n])]))[0];
      }
      return (await run('cond', [this.hb, this.prt, T('float32', Float32Array.of(1), [1])]))[0];
    }
    return (await run('cond', [this.hb, T('float32', new Float32Array(mt.D), [1, mt.D]), T('float32', Float32Array.of(0), [1])]))[0];
  }
  async push(cid) {
    const t = this.pos, lo = Math.max(0, (Math.floor(t / L) - 1) * L);
    if (lo > this.cstart) {                                    // новый блок окна букв: ключи раньше предыдущего блока не видны
      const cut = lo - this.cstart; this.enc = this.enc.map(c => rowsOf(c, cut, c.dims[2])); this.dec = this.dec.map(c => rowsOf(c, cut, c.dims[2])); this.cstart = lo;
    }
    const p = i64(t);
    const [e, ...enc] = await run('enc', [T('int64', BigInt64Array.from([BigInt(cid)]), [1, 1]), p, ...this.enc]); this.enc = enc;
    const is_sp = !!spacelike[cid], pos_in = t - this.lastb;
    const b_t = (is_sp && !this.prev_sp) || (pos_in % mt.max_word === 0 && pos_in > 0);
    this.word_e.push(e);
    if (!alpha[cid]) this.frag_cnt = 0;
    else {
      this.frag_cnt += 1;
      if (vowel[cid] && this.frag_cnt >= mt.frag_min) { this.frag_cnt = 0; if (!b_t) { this.hint_mark = this.word_e.length - 1; this.prt = null; this.cond = null; } }
    }
    if (b_t) {                                                 // слово закрылось на этой букве
      const E = cat1(this.word_e), n = E.dims[1];
      const pin = Float32Array.from({ length: n }, (_, i) => (i < n - 1 ? i + 1 : 0));
      const [wv] = await run('sread', [E, T('float32', pin, [n])]);
      const rot = (this.n_words + 1) % CHUNK === 0;
      const [hb, ...st] = await run('body', [wv, i64(this.n_words), T('bool', Uint8Array.of(rot ? 1 : 0), [1]), ...this.body]);
      this.body = rot ? st : st.map((x, i) => (mt.body.state[i][0] === 'S' ? this.body[i] : x));   // без переноса S тот же — держим прежний
      this.hb = hb; this.n_words += 1; this.lastb = t; this.word_e = []; this.hint_mark = -1; this.prt = null; this.cond = null;
    }
    if (this.cond === null || (this.hint_mark >= 0 && this.prt === null)) this.cond = await this._cond();
    const [logits, zn, ...dec] = await run('dec', [e, this.cond, p, ...this.dec]); this.dec = dec;
    this.zn = zn; this.prev_sp = is_sp; this.pos += 1;
    return logits.data;
  }
}

class CopyIndex {                                             // точные повторы для головы E, с откатом
  constructor() { this.x = []; this.last = SCALES.map(() => new Map()); this.cnt = SCALES.map(() => new Map()); this.undo = []; }
  push(c) {
    this.x.push(c); const t = this.x.length - 1, rec = [], cand = new Array(SCALES.length).fill(-1), feats = new Float32Array(SCALES.length * NF);
    SCALES.forEach((n, ki) => {
      if (t < n - 1) return;
      const key = this.x.slice(t - n + 1, t + 1).join(','), j = this.last[ki].get(key), r = this.cnt[ki].get(key) || 0;
      rec.push([ki, key, j, r]);
      if (j !== undefined) { cand[ki] = this.x[j + 1]; feats[ki * NF] = n / 24; feats[ki * NF + 1] = Math.log1p(r) / 5; feats[ki * NF + 2] = (t - j) / 4096; }
      this.cnt[ki].set(key, r + 1); this.last[ki].set(key, t);
    });
    this.undo.push(rec); return [cand, feats];
  }
  pop() {
    this.x.pop();
    for (const [ki, key, j, r] of this.undo.pop()) { if (j === undefined) { this.last[ki].delete(key); this.cnt[ki].delete(key); } else { this.last[ki].set(key, j); this.cnt[ki].set(key, r); } }
  }
}

async function mixCopy(zn, cf, logits) {                      // смесь с головой E → лог-вероятности (Float64Array)
  const lp = logSoftmax(logits);
  if (!cf) return lp;
  const [cand, feats] = cf; if (!cand.some(c => c >= 0)) return lp;
  const valid = Uint8Array.from(cand, c => (c >= 0 ? 1 : 0));
  const [w, a] = await run('copy', [zn, T('int64', BigInt64Array.from(cand, c => BigInt(Math.max(c, 0))), [cand.length]), T('float32', feats, [cand.length, NF]), T('bool', valid, [cand.length])]);
  const g = 1 / (1 + Math.exp(-a.data[0])), pc = new Float64Array(lp.length);
  cand.forEach((c, k) => { if (valid[k]) pc[Math.max(c, 0)] += w.data[k]; });
  const out = new Float64Array(lp.length); for (let i = 0; i < lp.length; i++) out[i] = Math.log(Math.max(g * pc[i] + (1 - g) * Math.exp(lp[i]), 1e-12));
  return out;
}

class Session {                                               // состояние поля: досылка, откат при правке, пересборка
  constructor() { this.reset([]); }
  async reset(ids) {
    const feed = ids.slice(ids.length - Math.min(ids.length, CTX)); this.ids = ids.slice(0, ids.length - feed.length); this.base = this.ids.length;
    this.r = new Runner(); this.ix = new CopyIndex(); this.lg = null; this.cf = null;
    this.archive = new Map(); this.words = [this.base]; this.ring = new Map([[this.base, this._pack()]]);
    for (const c of feed) await this._feed(c);
  }
  _pack() { const s = this.r.snapshot(), rows = [s.enc[0].dims[2], s.dec[0].dims[2], s.cstart]; delete s.enc; delete s.dec; return [s, rows, this.lg, this.cf]; }
  async _feed(c) {
    const r = this.r, cs = r.cstart, enc0 = r.enc, dec0 = r.dec, nw = r.n_words;
    this.lg = await r.push(c);
    if (r.cstart !== cs) this.archive.set(cs, [enc0, dec0]);
    this.cf = this.ix.push(c); this.ids.push(c); const p = this.ids.length; this.ring.set(p, this._pack());
    if (r.n_words !== nw) {
      this.words.push(p);
      if (this.words.length > KEEP_WORDS) {
        const lim = this.words[this.words.length - KEEP_WORDS - 1];
        for (const q of [...this.ring.keys()]) if (q < lim) this.ring.delete(q);
        this.words = this.words.slice(-KEEP_WORDS - 1);
        let keep = Infinity; for (const e of this.ring.values()) keep = Math.min(keep, e[1][2]);
        for (const st of [...this.archive.keys()]) if (st < keep) this.archive.delete(st);
      }
    }
  }
  _cut(n, start, which) {
    let cur = which === 0 ? this.r.enc : this.r.dec;
    if (this.r.cstart !== start) { if (!this.archive.has(start)) return null; cur = this.archive.get(start)[which]; }
    if (cur[0].dims[2] < n) return null;
    return cur.map(c => (c.dims[2] === n ? c : rowsOf(c, 0, n)));
  }
  _rollback(c) {
    const [s, [ne, nd, start], lg, cf] = this.ring.get(c), enc = this._cut(ne, start, 0), dec = this._cut(nd, start, 1);
    if (!enc || !dec) return false;
    for (let k = this.ids.length - c; k > 0; k--) this.ix.pop();
    this.r.restore(s); this.r.enc = enc; this.r.dec = dec; this.lg = lg; this.cf = cf; this.ids.length = c;
    for (const q of [...this.ring.keys()]) if (q > c) this.ring.delete(q);
    this.words = this.words.filter(q => q <= c); if (!this.words.length) this.words = [this.base];
    for (const st of [...this.archive.keys()]) if (st >= start) this.archive.delete(st);
    return true;
  }
  async sync(ids) {                                            // → лог-вероятности следующей буквы (со смесью E) или null
    const n = this.ids.length; let c = 0; const lim = Math.min(n, ids.length);
    while (c < lim && ids[c] === this.ids[c]) c++;
    if (c < n) {
      const ok = this.ring.has(c) && c - this.base >= Math.min(c, CTX >> 1) && this._rollback(c);
      if (!ok) { await this.reset(ids); return this._lp(); }
    }
    if (ids.length - this.ids.length > CTX || ids.length - this.base > RESET_AT) await this.reset(ids);
    else for (const ch of ids.slice(this.ids.length)) await this._feed(ch);
    return this._lp();
  }
  async _lp() { return this.lg === null ? null : mixCopy(this.r.zn, this.cf, this.lg); }
}

function topk(lp, k) {
  const idx = [];
  for (let i = 0; i < lp.length; i++) {
    if (idx.length < k) { idx.push(i); idx.sort((a, b) => lp[b] - lp[a]); }
    else if (lp[i] > lp[idx[k - 1]]) { idx[k - 1] = i; idx.sort((a, b) => lp[b] - lp[a]); }
  }
  return idx;
}

function branchCf(ix, rest, i) {                             // признаки E для буквы i на ветке луча: ветку временно дописать в индекс и снять журналом
  for (const c of rest) ix.push(c);
  const cf = ix.push(i);
  for (let k = 0; k <= rest.length; k++) ix.pop();
  return cf;
}

async function complete(s, lp) {                              // луч по буквам до первой небуквы → [[остаток, log p]] по убыванию
  const r = s.r, st0 = r.snapshot(); let beams = [[0, [], st0, lp]]; const done = [];
  try {
    for (let step = 0; step < MAXLEN; step++) {
      const cand = [];
      for (const [sc, rest, st, lpb] of beams)
        for (const i of topk(lpb, BEAM)) {                     // счёт кандидата известен до шага модели — шагаем только лучшие BEAM
          if (i === mt.eos) continue;
          if (!alpha[i]) { done.push([rest, sc + lpb[i]]); continue; }
          cand.push([sc + lpb[i], rest, st, i]);
        }
      cand.sort((a, b) => b[0] - a[0]); const nxt = [];
      for (const [sc, rest, st, i] of cand.slice(0, BEAM)) {
        r.restore(st); const lg = await r.push(i), st2 = r.snapshot();
        nxt.push([sc, rest.concat([i]), st2, await mixCopy(r.zn, branchCf(s.ix, rest, i), lg)]);
      }
      beams = nxt;
      if (!beams.length || (done.length && Math.max(...done.map(d => d[1])) > beams[0][0])) break;
    }
  } finally { r.restore(st0); }
  return done.sort((a, b) => b[1] - a[1]);
}

const idsOf = text => { const out = []; for (const ch of text) { const i = ch2id.get(ch); if (i !== undefined) out.push(i); } return out; };
const txt = ids => ids.map(i => mt.chars[i - mt.char0]).join('');

async function answer(text) {
  const chars = [...text];
  if (!chars.length || !ch2id.has(chars[chars.length - 1])) return { suggestions: [], after_space: false, word_end: false, ms_model: 0 };   // последний знак модели не виден
  const t0 = performance.now(), ids = idsOf(text), lp = await sess.sync(ids), after_space = !alpha[ids[ids.length - 1]];
  const res = lp ? await complete(sess, lp) : [];
  const word_end = res.length > 0 && res[0][0].length === 0, sugg = [], seen = new Set();
  if (!word_end) for (const [rest, l] of res) { const w = txt(rest); if (!w || seen.has(w)) continue; seen.add(w); sugg.push({ word: w, logp: l }); if (sugg.length === TOP) break; }
  return { suggestions: sugg, after_space, word_end, ms_model: Math.round(performance.now() - t0) };
}

let queue = Promise.resolve();                                // сообщения обрабатываются строго по очереди
onmessage = ev => {
  const m = ev.data;
  queue = queue.then(async () => {
    try {
      if (m.cmd === 'init') await init(m.base || 'model/', m.threads || 1, m.body);
      else if (m.cmd === 'complete') postMessage({ ev: 'answer', id: m.id, text: m.text, ...(await answer(m.text)) });
      else if (m.cmd === 'trace') {                           // для тестов: лог-вероятность каждой следующей буквы текста (без E) и последняя строка (с E)
        const ids = idsOf(m.text), r = new Runner(), each = []; let lg = null;
        for (let i = 0; i < ids.length; i++) { lg = await r.push(ids[i]); if (i + 1 < ids.length) each.push(logSoftmax(lg)[ids[i + 1]]); }
        const s = new Session(); await s.reset([]); const last = Array.from(await s.sync(ids));
        postMessage({ ev: 'trace', id: m.id, each, last, words: r.n_words });
      }
    } catch (e) { postMessage({ ev: 'error', id: m.id, error: String(e && e.stack || e) }); }
  });
};
