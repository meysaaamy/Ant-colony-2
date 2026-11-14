/*
 * BEST-OF SOURCE FILE
 * Source zip: neuroevolutionary-insect-simulation-engine (1).zip (neuroevolutionary-insect-simulation-engine__1_/src/lib/pheromone-grid.ts)
 * Why best: پد-stride + wsum (حلقه داخلی سریع‌تر)، mass-tracking فی کانال، tone-LUT ۱۰۲۴δ، PheromoneParams، CatmullRomBrush état‌دار، mixSample، DirtyRect
 * Borrow from other zips: fable: mixer با smooth() و renderRGBA
 * Notes: مرجع بهینه‌سازی عملکرد پ emission — در best-of/refs/
 * Status: verbatim copy — adapt import paths when merging into app/
 */

/*
 * 5-channel ping-pong pheromone field (cellSize = 6) with:
 *  - strict CFL 3x3 diffusion with zero-flux Neumann walls (walls hold 0, open-neighbour weight sum)
 *  - per-channel evaporation (alarm 3.2x faster), food+home polymer (35% slower decay)
 *  - chemical reaction: noEntry dissolves foodTrail (C_food -= 0.45 * C_noEntry * dt)
 *  - concentration-weighted spectral RGB mixer with interference terms
 *  - channel ablation masks and a Catmull-Rom brush (stamp spacing 0.25 * r)
 */
import { CHANNEL_COUNT, CH_ALARM, CH_FOOD, CH_HOME, CH_NOENTRY, CH_SCOUT, type PheromoneParams } from "./types";

export const PHEROMONE_CELL = 6;
export const PHEROMONE_MAX = 100;
export const CFL_MAX = 0.24;
const INV6 = 1 / 6;
/** Per-channel evaporation multipliers: food, home, alarm (3.2x), scout, noEntry */
export const DECAY_MULT: readonly number[] = [1, 0.8, 3.2, 1.6, 0.55];
const TONE_LUT_SIZE = 1024;
const TONE_LUT_MAX = 8;

export interface DirtyRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export class PheromoneGrid {
  readonly cellSize = PHEROMONE_CELL;
  readonly cols: number;
  readonly rows: number;
  readonly stride: number;
  readonly size: number;
  readonly worldW: number;
  readonly worldH: number;
  front: Float32Array[] = [];
  back: Float32Array[] = [];
  readonly wall: Uint8Array;
  readonly wsum: Float32Array;
  readonly muted = new Uint8Array(CHANNEL_COUNT);
  readonly mass = new Float32Array(CHANNEL_COUNT);
  private readonly touched = new Uint8Array(CHANNEL_COUNT);
  params: PheromoneParams = { evaporation: 0.07, diffusion: 0.35, reactionRate: 1 };
  gamma = 2.6;
  private readonly toneLut = new Float32Array(TONE_LUT_SIZE + 1);

  constructor(worldW: number, worldH: number) {
    this.worldW = worldW;
    this.worldH = worldH;
    this.cols = Math.ceil(worldW / PHEROMONE_CELL);
    this.rows = Math.ceil(worldH / PHEROMONE_CELL);
    this.stride = this.cols + 2;
    this.size = this.stride * (this.rows + 2);
    for (let c = 0; c < CHANNEL_COUNT; c++) {
      this.front.push(new Float32Array(this.size));
      this.back.push(new Float32Array(this.size));
    }
    this.wall = new Uint8Array(this.size);
    this.wsum = new Float32Array(this.size);
    for (let y = 0; y < this.rows + 2; y++) {
      for (let x = 0; x < this.stride; x++) {
        if (x === 0 || y === 0 || x === this.stride - 1 || y === this.rows + 1) this.wall[y * this.stride + x] = 1;
      }
    }
    this.recomputeWsum(0, 0, this.cols + 1, this.rows + 1);
    this.buildToneLut();
  }

  buildToneLut(): void {
    for (let i = 0; i <= TONE_LUT_SIZE; i++) {
      const x = (i / TONE_LUT_SIZE) * TONE_LUT_MAX;
      this.toneLut[i] = 1 - Math.exp(-this.gamma * x);
    }
  }

  /** Interior cell index for world coordinates (clamped into the padded grid). */
  index(x: number, y: number): number {
    let cx = (x / PHEROMONE_CELL) | 0;
    let cy = (y / PHEROMONE_CELL) | 0;
    if (cx < 0) cx = 0;
    else if (cx >= this.cols) cx = this.cols - 1;
    if (cy < 0) cy = 0;
    else if (cy >= this.rows) cy = this.rows - 1;
    return (cy + 1) * this.stride + cx + 1;
  }

  isWallAt(x: number, y: number): boolean {
    if (x < 0 || y < 0 || x >= this.worldW || y >= this.worldH) return true;
    return this.wall[this.index(x, y)] === 1;
  }

  sample(ch: number, x: number, y: number): number {
    if (this.muted[ch] === 1) return 0;
    return this.front[ch][this.index(x, y)];
  }

  deposit(ch: number, x: number, y: number, amount: number): void {
    if (this.muted[ch] === 1 || amount <= 0) return;
    const i = this.index(x, y);
    if (this.wall[i] === 1) return;
    const a = this.front[ch];
    const v = a[i] + amount;
    a[i] = v > PHEROMONE_MAX ? PHEROMONE_MAX : v;
    this.touched[ch] = 1;
  }

  /** Soft circular stamp with quadratic falloff. */
  depositRadius(ch: number, x: number, y: number, r: number, amount: number): void {
    if (this.muted[ch] === 1) return;
    const cr = Math.max(1, Math.ceil(r / PHEROMONE_CELL));
    const cx = (x / PHEROMONE_CELL) | 0;
    const cy = (y / PHEROMONE_CELL) | 0;
    const a = this.front[ch];
    for (let dy = -cr; dy <= cr; dy++) {
      const gy = cy + dy;
      if (gy < 0 || gy >= this.rows) continue;
      for (let dx = -cr; dx <= cr; dx++) {
        const gx = cx + dx;
        if (gx < 0 || gx >= this.cols) continue;
        const d2 = (dx * dx + dy * dy) / (cr * cr);
        if (d2 > 1) continue;
        const i = (gy + 1) * this.stride + gx + 1;
        if (this.wall[i] === 1) continue;
        const v = a[i] + amount * (1 - d2);
        a[i] = v > PHEROMONE_MAX ? PHEROMONE_MAX : v < 0 ? 0 : v;
      }
    }
    this.touched[ch] = 1;
  }

  /** Erase all channels inside a radius. */
  eraseRadius(x: number, y: number, r: number): void {
    const cr = Math.max(1, Math.ceil(r / PHEROMONE_CELL));
    const cx = (x / PHEROMONE_CELL) | 0;
    const cy = (y / PHEROMONE_CELL) | 0;
    for (let dy = -cr; dy <= cr; dy++) {
      const gy = cy + dy;
      if (gy < 0 || gy >= this.rows) continue;
      for (let dx = -cr; dx <= cr; dx++) {
        const gx = cx + dx;
        if (gx < 0 || gx >= this.cols || dx * dx + dy * dy > cr * cr) continue;
        const i = (gy + 1) * this.stride + gx + 1;
        for (let c = 0; c < CHANNEL_COUNT; c++) this.front[c][i] *= 0.15;
      }
    }
  }

  /** Paint (on=true) or carve walls in a circle. Returns the dirty rect in world px. */
  setWallCircle(x: number, y: number, r: number, on: boolean): DirtyRect {
    const cr = Math.max(0.5, r / PHEROMONE_CELL);
    const cx = x / PHEROMONE_CELL;
    const cy = y / PHEROMONE_CELL;
    const gx0 = Math.max(0, Math.floor(cx - cr));
    const gx1 = Math.min(this.cols - 1, Math.ceil(cx + cr));
    const gy0 = Math.max(0, Math.floor(cy - cr));
    const gy1 = Math.min(this.rows - 1, Math.ceil(cy + cr));
    for (let gy = gy0; gy <= gy1; gy++) {
      for (let gx = gx0; gx <= gx1; gx++) {
        const dx = gx + 0.5 - cx;
        const dy = gy + 0.5 - cy;
        if (dx * dx + dy * dy > cr * cr) continue;
        const i = (gy + 1) * this.stride + gx + 1;
        this.wall[i] = on ? 1 : 0;
        if (on) for (let c = 0; c < CHANNEL_COUNT; c++) this.front[c][i] = 0;
      }
    }
    this.recomputeWsum(gx0, gy0, gx1 + 2, gy1 + 2);
    return { x0: gx0 * PHEROMONE_CELL, y0: gy0 * PHEROMONE_CELL, x1: (gx1 + 1) * PHEROMONE_CELL, y1: (gy1 + 1) * PHEROMONE_CELL };
  }

  clearWalls(): void {
    for (let y = 1; y <= this.rows; y++) for (let x = 1; x <= this.cols; x++) this.wall[y * this.stride + x] = 0;
    this.recomputeWsum(0, 0, this.cols + 1, this.rows + 1);
  }

  /** Recompute open-neighbour weight sums in padded coords [x0..x1]x[y0..y1]. */
  recomputeWsum(x0: number, y0: number, x1: number, y1: number): void {
    const s = this.stride;
    const ax0 = Math.max(1, x0);
    const ay0 = Math.max(1, y0);
    const ax1 = Math.min(this.cols, x1);
    const ay1 = Math.min(this.rows, y1);
    for (let y = ay0; y <= ay1; y++) {
      for (let x = ax0; x <= ax1; x++) {
        const i = y * s + x;
        if (this.wall[i] === 1) {
          this.wsum[i] = 0;
          continue;
        }
        const w = this.wall;
        const orth = 4 - (w[i - 1] + w[i + 1] + w[i - s] + w[i + s]);
        const diag = 4 - (w[i - s - 1] + w[i - s + 1] + w[i + s - 1] + w[i + s + 1]);
        this.wsum[i] = (orth + 0.5 * diag) * INV6;
      }
    }
  }

  clear(ch?: number): void {
    for (let c = 0; c < CHANNEL_COUNT; c++) {
      if (ch !== undefined && ch !== c) continue;
      this.front[c].fill(0);
      this.back[c].fill(0);
      this.mass[c] = 0;
    }
  }

  setMuted(ch: number, muted: boolean): void {
    this.muted[ch] = muted ? 1 : 0;
    if (muted) this.clear(ch);
  }

  /** Reaction-diffusion step. dt in seconds. */
  step(dt: number): void {
    const { evaporation, diffusion, reactionRate } = this.params;
    const k = Math.min(CFL_MAX, Math.max(0, diffusion * CFL_MAX * dt * 60));
    const s = this.stride;
    const wall = this.wall;
    const wsum = this.wsum;
    const F = this.front[CH_FOOD];
    const H = this.front[CH_HOME];
    const N = this.front[CH_NOENTRY];
    const dissolve = 0.45 * reactionRate * dt;
    for (let ch = 0; ch < CHANNEL_COUNT; ch++) {
      if (this.mass[ch] <= 0 && this.touched[ch] === 0) continue;
      this.touched[ch] = 0;
      const src = this.front[ch];
      const dst = this.back[ch];
      const baseDecay = evaporation * DECAY_MULT[ch] * dt;
      const isFood = ch === CH_FOOD;
      const isPolymer = ch === CH_FOOD || ch === CH_HOME;
      // numerical floor only when the channel is actually evaporating (keeps pure diffusion exactly conservative)
      const floor = baseDecay > 0 ? 0.004 : 0;
      let mass = 0;
      for (let y = 1; y <= this.rows; y++) {
        let i = y * s + 1;
        for (let x = 1; x <= this.cols; x++, i++) {
          if (wall[i] === 1) {
            dst[i] = 0;
            continue;
          }
          const c = src[i];
          const orth = src[i - 1] + src[i + 1] + src[i - s] + src[i + s];
          const diag = src[i - s - 1] + src[i - s + 1] + src[i + s - 1] + src[i + s + 1];
          let v = c + k * ((orth + 0.5 * diag) * INV6 - wsum[i] * c);
          let decay = baseDecay;
          if (isPolymer) {
            const ov = F[i] < H[i] ? F[i] : H[i];
            if (ov > 0.5) decay *= 1 - 0.35 * (ov > 25 ? 1 : ov / 25);
          }
          v *= 1 - decay;
          if (isFood) v -= dissolve * N[i];
          if (v < floor || v < 0) v = 0;
          else if (v > PHEROMONE_MAX) v = PHEROMONE_MAX;
          dst[i] = v;
          mass += v;
        }
      }
      this.mass[ch] = mass;
      this.front[ch] = dst;
      this.back[ch] = src;
    }
  }

  /**
   * Concentration-weighted spectral mixer -> RGBA (cols x rows), tonemapped 1 - exp(-gamma * sum).
   * Interference: food+home => turquoise edges / pearl cores, food+scout => rose-gold coral,
   * home+alarm => electric magenta, alarm+noEntry => sulfur gold.
   */
  mixToRGBA(out: Uint8ClampedArray, intensity: number): void {
    const F = this.front[CH_FOOD];
    const H = this.front[CH_HOME];
    const A = this.front[CH_ALARM];
    const S = this.front[CH_SCOUT];
    const N = this.front[CH_NOENTRY];
    const lut = this.toneLut;
    const scale = TONE_LUT_SIZE / TONE_LUT_MAX;
    const k = 0.01 * intensity;
    let p = 0;
    for (let y = 1; y <= this.rows; y++) {
      let i = y * this.stride + 1;
      for (let x = 1; x <= this.cols; x++, i++) {
        const f = F[i] * k;
        const h = H[i] * k;
        const a = A[i] * k;
        const sc = S[i] * k;
        const n = N[i] * k;
        if (f + h + a + sc + n < 0.002) {
          out[p] = 0;
          out[p + 1] = 0;
          out[p + 2] = 0;
          out[p + 3] = 255;
          p += 4;
          continue;
        }
        let r = f + a + 0.686 * sc + 0.314 * n;
        let g = 0.659 * f + 0.882 * h + 0.047 * a + 0.196 * sc + n;
        let b = h + 0.294 * a + sc + 0.392 * n;
        const fh = Math.sqrt(f * h);
        if (fh > 0.002) {
          const core = fh * fh * 2.2;
          const edge = fh * (1 - (fh > 1 ? 1 : fh)) * 1.5;
          r += core * 1.0 + edge * 0.25;
          g += core * 0.97 + edge * 1.0;
          b += core * 0.92 + edge * 0.85;
        }
        const fs = Math.sqrt(f * sc);
        if (fs > 0.002) {
          r += fs * 1.3;
          g += fs * 0.62;
          b += fs * 0.5;
        }
        const ha = Math.sqrt(h * a);
        if (ha > 0.002) {
          r += ha * 1.4;
          g += ha * 0.1;
          b += ha * 1.25;
        }
        const an = Math.sqrt(a * n);
        if (an > 0.002) {
          r += an * 1.3;
          g += an * 1.15;
          b += an * 0.12;
        }
        let ri = (r * scale) | 0;
        let gi = (g * scale) | 0;
        let bi = (b * scale) | 0;
        if (ri > TONE_LUT_SIZE) ri = TONE_LUT_SIZE;
        if (gi > TONE_LUT_SIZE) gi = TONE_LUT_SIZE;
        if (bi > TONE_LUT_SIZE) bi = TONE_LUT_SIZE;
        out[p] = lut[ri] * 255;
        out[p + 1] = lut[gi] * 255;
        out[p + 2] = lut[bi] * 255;
        out[p + 3] = 255;
        p += 4;
      }
    }
  }

  /** Mix a single synthetic sample (used for previews & health checks). Returns [r,g,b] 0..255. */
  mixSample(c: readonly number[]): [number, number, number] {
    const f = c[0] * 0.01;
    const h = c[1] * 0.01;
    const a = c[2] * 0.01;
    const sc = c[3] * 0.01;
    const n = c[4] * 0.01;
    let r = f + a + 0.686 * sc + 0.314 * n;
    let g = 0.659 * f + 0.882 * h + 0.047 * a + 0.196 * sc + n;
    let b = h + 0.294 * a + sc + 0.392 * n;
    const fh = Math.sqrt(f * h);
    r += fh * fh * 2.2 + fh * (1 - Math.min(1, fh)) * 1.5 * 0.25;
    g += fh * fh * 2.2 * 0.97 + fh * (1 - Math.min(1, fh)) * 1.5;
    b += fh * fh * 2.2 * 0.92 + fh * (1 - Math.min(1, fh)) * 1.5 * 0.85;
    const fs = Math.sqrt(f * sc);
    r += fs * 1.3;
    g += fs * 0.62;
    b += fs * 0.5;
    const ha = Math.sqrt(h * a);
    r += ha * 1.4;
    g += ha * 0.1;
    b += ha * 1.25;
    const an = Math.sqrt(a * n);
    r += an * 1.3;
    g += an * 1.15;
    b += an * 0.12;
    const t = (v: number) => Math.round((1 - Math.exp(-this.gamma * v)) * 255);
    return [t(r), t(g), t(b)];
  }
}

/**
 * Catmull-Rom brush: feed raw pointer samples, receive evenly spaced stamps (spacing = 0.25 * r).
 * Fixed 4-point ring, no allocation per sample.
 */
export class CatmullRomBrush {
  private px = new Float32Array(4);
  private py = new Float32Array(4);
  private count = 0;
  private carry = 0;

  reset(): void {
    this.count = 0;
    this.carry = 0;
  }

  /** Returns number of stamps emitted through `stamp`. */
  add(x: number, y: number, radius: number, stamp: (sx: number, sy: number) => void): number {
    const px = this.px;
    const py = this.py;
    if (this.count === 0) {
      for (let i = 0; i < 4; i++) {
        px[i] = x;
        py[i] = y;
      }
      this.count = 1;
      stamp(x, y);
      return 1;
    }
    px[0] = px[1];
    py[0] = py[1];
    px[1] = px[2];
    py[1] = py[2];
    px[2] = px[3];
    py[2] = py[3];
    px[3] = x;
    py[3] = y;
    this.count++;
    // interpolate between p1 and p2 (one-sample latency)
    const spacing = Math.max(1, 0.25 * radius);
    const segLen = Math.hypot(px[2] - px[1], py[2] - py[1]);
    if (segLen < 1e-3) return 0;
    let emitted = 0;
    let t = (spacing - this.carry) / segLen;
    while (t <= 1) {
      const t2 = t * t;
      const t3 = t2 * t;
      const sx = 0.5 * (2 * px[1] + (-px[0] + px[2]) * t + (2 * px[0] - 5 * px[1] + 4 * px[2] - px[3]) * t2 + (-px[0] + 3 * px[1] - 3 * px[2] + px[3]) * t3);
      const sy = 0.5 * (2 * py[1] + (-py[0] + py[2]) * t + (2 * py[0] - 5 * py[1] + 4 * py[2] - py[3]) * t2 + (-py[0] + 3 * py[1] - 3 * py[2] + py[3]) * t3);
      stamp(sx, sy);
      emitted++;
      t += spacing / segLen;
    }
    this.carry = (1 - (t - spacing / segLen)) * segLen;
    if (this.carry > spacing) this.carry = 0;
    return emitted;
  }

  /** Flush the final segment to the last point. */
  end(radius: number, stamp: (sx: number, sy: number) => void): void {
    if (this.count > 1) this.add(this.px[3], this.py[3], radius, stamp);
    this.reset();
  }
}
