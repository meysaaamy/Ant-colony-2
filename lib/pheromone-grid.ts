/**
 * 5-channel ping-pong pheromone field.
 *  - Float32Array grids, cellSize = 6 world px, concentrations clamped to [0, 100]
 *  - explicit 3x3 diffusion under a strict CFL clamp with zero-flux (Neumann) walls
 *  - reaction terms: foodTrail+homeGuide polymer (35% slower decay), noEntry dissolves foodTrail
 *  - concentration-weighted spectral RGB mixer with interference colours
 *  - channel ablation masks (sensing mutes) and a Catmull-Rom brush sampler
 */
import { CHANNEL_COUNT, CHANNEL_RGB, CH_ALARM, CH_FOOD, CH_HOME, CH_NOENTRY } from './types';

export const PHEROMONE_CELL = 6;
export const PHEROMONE_MAX = 100;
/** Explicit 3x3 stencil is a convex blend (hence stable) while alpha <= 1; we stay well inside it. */
export const CFL_LIMIT = 0.8;
export const ALARM_DECAY_MULT = 3.2;
export const POLYMER_SLOWDOWN = 0.35;
export const NOENTRY_DISSOLVE = 0.45;

/** Base evaporation rates in 1/s (alarm is 3.2x the foodTrail rate). */
export const BASE_EVAPORATION: readonly number[] = [0.06, 0.03, 0.06 * ALARM_DECAY_MULT, 0.12, 0.08];

const W_ORTHO = 0.2;
const W_DIAG = 0.05;
const POLYMER_K = 6;
const FLOOR = 0.002;

export class PheromoneGrid {
  readonly width: number;
  readonly height: number;
  readonly cols: number;
  readonly rows: number;
  readonly size: number;
  readonly cellSize = PHEROMONE_CELL;

  cur: Float32Array[] = [];
  nxt: Float32Array[] = [];
  /** 1 = basalt wall (shared with the engine for collisions). */
  readonly wall: Uint8Array;
  /** 1 = cell touches the border or a wall and needs the slow Neumann stencil. */
  readonly edge: Uint8Array;
  /** Channel ablation masks: 0 mutes sensing (and display) of a channel. */
  readonly mask = new Float32Array(CHANNEL_COUNT).fill(1);
  readonly active = new Uint8Array(CHANNEL_COUNT);

  /** Global evaporation multiplier. */
  evaporation = 1;
  /** Diffusion coefficient in cells^2 / s (pre-CFL clamp). */
  diffusion = 2;
  /** Multiplier on cross-reactions (polymer + noEntry dissolve). */
  crossReaction = 1;
  wallCount = 0;
  wallVersion = 0;
  /** Last alpha actually used by step(): exposed so tests can assert the CFL bound. */
  lastAlpha = 0;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.cols = Math.ceil(width / PHEROMONE_CELL);
    this.rows = Math.ceil(height / PHEROMONE_CELL);
    this.size = this.cols * this.rows;
    for (let c = 0; c < CHANNEL_COUNT; c++) {
      this.cur.push(new Float32Array(this.size));
      this.nxt.push(new Float32Array(this.size));
    }
    this.wall = new Uint8Array(this.size);
    this.edge = new Uint8Array(this.size);
    this.rebuildEdges(0, 0, this.cols - 1, this.rows - 1);
  }

  /* ---------------------------- indexing ---------------------------- */

  cellIndex(x: number, y: number): number {
    if (x < 0 || y < 0) return -1;
    const gx = (x / PHEROMONE_CELL) | 0;
    const gy = (y / PHEROMONE_CELL) | 0;
    if (gx >= this.cols || gy >= this.rows) return -1;
    return gy * this.cols + gx;
  }

  isWallAt(x: number, y: number): boolean {
    const i = this.cellIndex(x, y);
    return i < 0 || this.wall[i] !== 0;
  }

  sample(ch: number, x: number, y: number): number {
    const i = this.cellIndex(x, y);
    return i < 0 ? 0 : this.cur[ch][i];
  }

  deposit(ch: number, x: number, y: number, amount: number): void {
    const i = this.cellIndex(x, y);
    if (i < 0 || this.wall[i] !== 0) return;
    const g = this.cur[ch];
    const v = g[i] + amount;
    g[i] = v > PHEROMONE_MAX ? PHEROMONE_MAX : v;
    this.active[ch] = 1;
  }

  /* ---------------------------- editing ----------------------------- */

  /** Soft disc stamp (quadratic falloff) used by the brush. */
  stampDisc(ch: number, x: number, y: number, r: number, amount: number): void {
    const cs = PHEROMONE_CELL;
    const x0 = Math.max(0, Math.floor((x - r) / cs));
    const x1 = Math.min(this.cols - 1, Math.floor((x + r) / cs));
    const y0 = Math.max(0, Math.floor((y - r) / cs));
    const y1 = Math.min(this.rows - 1, Math.floor((y + r) / cs));
    const g = this.cur[ch];
    const inv = 1 / (r * r);
    for (let gy = y0; gy <= y1; gy++) {
      for (let gx = x0; gx <= x1; gx++) {
        const dx = (gx + 0.5) * cs - x;
        const dy = (gy + 0.5) * cs - y;
        const d2 = (dx * dx + dy * dy) * inv;
        if (d2 >= 1) continue;
        const i = gy * this.cols + gx;
        if (this.wall[i] !== 0) continue;
        const v = g[i] + amount * (1 - d2);
        g[i] = v > PHEROMONE_MAX ? PHEROMONE_MAX : v;
      }
    }
    this.active[ch] = 1;
  }

  eraseDisc(x: number, y: number, r: number): void {
    const cs = PHEROMONE_CELL;
    const x0 = Math.max(0, Math.floor((x - r) / cs));
    const x1 = Math.min(this.cols - 1, Math.floor((x + r) / cs));
    const y0 = Math.max(0, Math.floor((y - r) / cs));
    const y1 = Math.min(this.rows - 1, Math.floor((y + r) / cs));
    for (let gy = y0; gy <= y1; gy++) {
      for (let gx = x0; gx <= x1; gx++) {
        const dx = (gx + 0.5) * cs - x;
        const dy = (gy + 0.5) * cs - y;
        if (dx * dx + dy * dy >= r * r) continue;
        const i = gy * this.cols + gx;
        for (let c = 0; c < CHANNEL_COUNT; c++) this.cur[c][i] = 0;
      }
    }
  }

  /** Paint or carve wall cells inside a disc. Returns the number of cells changed. */
  setWallDisc(x: number, y: number, r: number, solid: boolean): number {
    const cs = PHEROMONE_CELL;
    const x0 = Math.max(0, Math.floor((x - r) / cs));
    const x1 = Math.min(this.cols - 1, Math.floor((x + r) / cs));
    const y0 = Math.max(0, Math.floor((y - r) / cs));
    const y1 = Math.min(this.rows - 1, Math.floor((y + r) / cs));
    const v = solid ? 1 : 0;
    let changed = 0;
    for (let gy = y0; gy <= y1; gy++) {
      for (let gx = x0; gx <= x1; gx++) {
        const dx = (gx + 0.5) * cs - x;
        const dy = (gy + 0.5) * cs - y;
        if (dx * dx + dy * dy > r * r) continue;
        const i = gy * this.cols + gx;
        if (this.wall[i] === v) continue;
        this.wall[i] = v;
        this.wallCount += solid ? 1 : -1;
        changed++;
        if (solid) for (let c = 0; c < CHANNEL_COUNT; c++) this.cur[c][i] = 0;
      }
    }
    if (changed > 0) {
      this.rebuildEdges(x0 - 1, y0 - 1, x1 + 1, y1 + 1);
      this.wallVersion++;
    }
    return changed;
  }

  rebuildEdges(x0: number, y0: number, x1: number, y1: number): void {
    const cols = this.cols;
    const rows = this.rows;
    const wall = this.wall;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > cols - 1) x1 = cols - 1;
    if (y1 > rows - 1) y1 = rows - 1;
    for (let gy = y0; gy <= y1; gy++) {
      for (let gx = x0; gx <= x1; gx++) {
        const i = gy * cols + gx;
        let e = 0;
        if (gx === 0 || gy === 0 || gx === cols - 1 || gy === rows - 1) e = 1;
        else if (
          wall[i - 1] | wall[i + 1] | wall[i - cols] | wall[i + cols] |
          wall[i - cols - 1] | wall[i - cols + 1] | wall[i + cols - 1] | wall[i + cols + 1]
        ) e = 1;
        this.edge[i] = e;
      }
    }
  }

  clearPheromones(): void {
    for (let c = 0; c < CHANNEL_COUNT; c++) {
      this.cur[c].fill(0);
      this.nxt[c].fill(0);
      this.active[c] = 0;
    }
  }

  clearWalls(): void {
    this.wall.fill(0);
    this.wallCount = 0;
    this.wallVersion++;
    this.rebuildEdges(0, 0, this.cols - 1, this.rows - 1);
  }

  setMask(ch: number, enabled: boolean): void {
    this.mask[ch] = enabled ? 1 : 0;
  }

  /* --------------------------- simulation --------------------------- */

  private lapSlow(src: Float32Array, gx: number, gy: number, c: number): number {
    const cols = this.cols;
    const rows = this.rows;
    const wall = this.wall;
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const w = dx !== 0 && dy !== 0 ? W_DIAG : W_ORTHO;
        const nx = gx + dx;
        const ny = gy + dy;
        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) {
          s += w * c; // zero-flux: mirror the centre value
          continue;
        }
        const j = ny * cols + nx;
        s += w * (wall[j] !== 0 ? c : src[j]);
      }
    }
    return s - c;
  }

  /** Reaction-diffusion step. dt in seconds. */
  step(dt: number): void {
    const cols = this.cols;
    const rows = this.rows;
    const wall = this.wall;
    const edge = this.edge;
    let alpha = this.diffusion * dt;
    if (alpha > CFL_LIMIT) alpha = CFL_LIMIT;
    if (alpha < 0) alpha = 0;
    this.lastAlpha = alpha;
    const food0 = this.cur[CH_FOOD];
    const home0 = this.cur[CH_HOME];
    const noEntry0 = this.cur[CH_NOENTRY];
    const cross = this.crossReaction;
    const polymer = POLYMER_SLOWDOWN * (cross > 1 ? 1 : cross);
    const dissolve = NOENTRY_DISSOLVE * cross * dt;
    const noEntryActive = this.active[CH_NOENTRY] !== 0;

    for (let ch = 0; ch < CHANNEL_COUNT; ch++) {
      if (this.active[ch] === 0) continue;
      const src = this.cur[ch];
      const dst = this.nxt[ch];
      let rate = BASE_EVAPORATION[ch] * this.evaporation * dt;
      if (rate > 0.95) rate = 0.95;
      const isPolymer = ch === CH_FOOD || ch === CH_HOME;
      const isFood = ch === CH_FOOD && noEntryActive;
      let maxV = 0;
      for (let gy = 0; gy < rows; gy++) {
        let i = gy * cols;
        for (let gx = 0; gx < cols; gx++, i++) {
          if (wall[i] !== 0) {
            dst[i] = 0;
            continue;
          }
          const c = src[i];
          let lap: number;
          if (edge[i] === 0) {
            lap =
              W_ORTHO * (src[i - 1] + src[i + 1] + src[i - cols] + src[i + cols]) +
              W_DIAG * (src[i - cols - 1] + src[i - cols + 1] + src[i + cols - 1] + src[i + cols + 1]) -
              c;
          } else {
            lap = this.lapSlow(src, gx, gy, c);
          }
          let v = c + alpha * lap;
          if (v > 0) {
            let r = rate;
            if (isPolymer) {
              const f = food0[i];
              const h = home0[i];
              const m = f < h ? f : h;
              if (m > 0.05) r *= 1 - polymer * (m / (m + POLYMER_K));
            }
            v -= v * r;
            if (isFood) v -= dissolve * noEntry0[i];
            if (v < FLOOR) v = 0;
            else if (v > PHEROMONE_MAX) v = PHEROMONE_MAX;
          } else {
            v = 0;
          }
          dst[i] = v;
          if (v > maxV) maxV = v;
        }
      }
      if (maxV === 0) this.active[ch] = 0;
    }
    // ping-pong swap (inactive channels are all-zero on both sides)
    for (let ch = 0; ch < CHANNEL_COUNT; ch++) {
      const t = this.cur[ch];
      this.cur[ch] = this.nxt[ch];
      this.nxt[ch] = t;
      if (this.active[ch] === 0 && t !== this.cur[ch]) this.cur[ch].fill(0);
    }
  }

  /** Total mass of one channel (used by tests and the health probe). */
  total(ch: number): number {
    const g = this.cur[ch];
    let s = 0;
    for (let i = 0; i < g.length; i++) s += g[i];
    return s;
  }

  /* ---------------------------- rendering --------------------------- */

  /** Writes cols*rows RGBA pixels through the spectral mixer. Alpha is opaque: composite with 'screen'. */
  renderRGBA(out: Uint8ClampedArray, gamma: number): void {
    const f = this.cur[0];
    const h = this.cur[1];
    const a = this.cur[2];
    const s = this.cur[3];
    const n = this.cur[4];
    const m = this.mask;
    const m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3], m4 = m[4];
    const size = this.size;
    for (let i = 0, o = 0; i < size; i++, o += 4) {
      const cf = f[i] * m0;
      const chh = h[i] * m1;
      const ca = a[i] * m2;
      const csc = s[i] * m3;
      const cn = n[i] * m4;
      if (cf + chh + ca + csc + cn < 0.04) {
        out[o] = 0;
        out[o + 1] = 0;
        out[o + 2] = 0;
        out[o + 3] = 255;
        continue;
      }
      mixPheromoneRGB(cf, chh, ca, csc, cn, gamma, out, o);
      out[o + 3] = 255;
    }
  }
}

/* ------------------------------------------------------------------ */
/* Concentration-weighted spectral RGB mixer                           */
/* ------------------------------------------------------------------ */

/** Perceptual channel weights w_i. */
export const CHANNEL_WEIGHTS: readonly number[] = [1.0, 0.9, 1.25, 1.1, 0.95];
export const DEFAULT_GAMMA = 3.4;

const R0 = CHANNEL_RGB[0][0] / 255, G0 = CHANNEL_RGB[0][1] / 255, B0 = CHANNEL_RGB[0][2] / 255;
const R1 = CHANNEL_RGB[1][0] / 255, G1 = CHANNEL_RGB[1][1] / 255, B1 = CHANNEL_RGB[1][2] / 255;
const R2 = CHANNEL_RGB[2][0] / 255, G2 = CHANNEL_RGB[2][1] / 255, B2 = CHANNEL_RGB[2][2] / 255;
const R3 = CHANNEL_RGB[3][0] / 255, G3 = CHANNEL_RGB[3][1] / 255, B3 = CHANNEL_RGB[3][2] / 255;
const R4 = CHANNEL_RGB[4][0] / 255, G4 = CHANNEL_RGB[4][1] / 255, B4 = CHANNEL_RGB[4][2] / 255;

function smooth(e0: number, e1: number, x: number): number {
  let t = (x - e0) / (e1 - e0);
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  t = t * t * (3 - 2 * t);
  return t;
}

/**
 * rgb = 1 - exp(-gamma * sum_i(w_i * C_i * RGB_i)), then pairwise interference:
 *   food+home   -> turquoise edges, pearl-white highway cores
 *   food+scout  -> rose-gold / coral
 *   home+alarm  -> electric magenta escape plumes
 *   alarm+noEntry -> sulfur-gold hazard zones
 * Concentrations are raw [0, 100]. Writes 3 bytes at out[o..o+2].
 */
export function mixPheromoneRGB(
  food: number,
  home: number,
  alarm: number,
  scout: number,
  noEntry: number,
  gamma: number,
  out: Uint8ClampedArray | number[],
  o: number,
): void {
  const cf = food * 0.01 * CHANNEL_WEIGHTS[0];
  const ch = home * 0.01 * CHANNEL_WEIGHTS[1];
  const ca = alarm * 0.01 * CHANNEL_WEIGHTS[2];
  const cs = scout * 0.01 * CHANNEL_WEIGHTS[3];
  const cn = noEntry * 0.01 * CHANNEL_WEIGHTS[4];
  let r = 1 - Math.exp(-gamma * (cf * R0 + ch * R1 + ca * R2 + cs * R3 + cn * R4));
  let g = 1 - Math.exp(-gamma * (cf * G0 + ch * G1 + ca * G2 + cs * G3 + cn * G4));
  let b = 1 - Math.exp(-gamma * (cf * B0 + ch * B1 + ca * B2 + cs * B3 + cn * B4));

  const lum = r > g ? (r > b ? r : b) : g > b ? g : b;
  const body = lum * 1.3 + 0.06 > 1 ? 1 : lum * 1.3 + 0.06;

  // food + home: turquoise shoulders, pearl-white polymer core
  const fh = cf * ch;
  if (fh > 1e-5) {
    const p = Math.sqrt(fh);
    const core = smooth(0.1, 0.34, p);
    const edgeK = smooth(0.012, 0.08, p) * (1 - core) * 0.8;
    r += (0.18 * body - r) * edgeK;
    g += (0.92 * body - g) * edgeK;
    b += (0.84 * body - b) * edgeK;
    const cb = lum * 1.7 + 0.2 > 1 ? 1 : lum * 1.7 + 0.2;
    r += (0.97 * cb - r) * core;
    g += (0.985 * cb - g) * core;
    b += (1.0 * cb - b) * core;
  }
  // food + scout: rose-gold / coral
  const fs = cf * cs;
  if (fs > 1e-5) {
    const k = smooth(0.012, 0.2, Math.sqrt(fs)) * 0.85;
    r += (1.0 * body - r) * k;
    g += (0.52 * body - g) * k;
    b += (0.44 * body - b) * k;
  }
  // home + alarm: electric magenta
  const ha = ch * ca;
  if (ha > 1e-5) {
    const k = smooth(0.012, 0.2, Math.sqrt(ha)) * 0.9;
    r += (1.0 * body - r) * k;
    g += (0.04 * body - g) * k;
    b += (0.9 * body - b) * k;
  }
  // alarm + noEntry: sulfur-gold
  const an = ca * cn;
  if (an > 1e-5) {
    const k = smooth(0.012, 0.2, Math.sqrt(an)) * 0.9;
    r += (0.95 * body - r) * k;
    g += (0.86 * body - g) * k;
    b += (0.14 * body - b) * k;
  }
  out[o] = r <= 0 ? 0 : r >= 1 ? 255 : (r * 255 + 0.5) | 0;
  out[o + 1] = g <= 0 ? 0 : g >= 1 ? 255 : (g * 255 + 0.5) | 0;
  out[o + 2] = b <= 0 ? 0 : b >= 1 ? 255 : (b * 255 + 0.5) | 0;
}

/* ------------------------------------------------------------------ */
/* Catmull-Rom brush                                                   */
/* ------------------------------------------------------------------ */

function cr(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

/**
 * Samples the Catmull-Rom segment p1 -> p2 at roughly `spacing` px (callers pass 0.25 * radius).
 * Writes x,y pairs into `out` and returns the point count. Allocation free.
 */
export function sampleCatmullRom(
  x0: number, y0: number,
  x1: number, y1: number,
  x2: number, y2: number,
  x3: number, y3: number,
  spacing: number,
  out: Float32Array,
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const cap = out.length >> 1;
  let n = Math.ceil(dist / (spacing > 0.5 ? spacing : 0.5));
  if (n < 1) n = 1;
  if (n > cap) n = cap;
  for (let k = 0; k < n; k++) {
    const t = (k + 1) / n;
    out[k * 2] = cr(x0, x1, x2, x3, t);
    out[k * 2 + 1] = cr(y0, y1, y2, y3, t);
  }
  return n;
}
