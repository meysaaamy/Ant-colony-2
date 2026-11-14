/**
 * Deterministic randomness: Mulberry32 PRNG, XorShift32 hash step and seeded 2D Simplex noise.
 */

export class Rng {
  private s: number;

  constructor(seed: number) {
    this.s = seed >>> 0;
  }

  reseed(seed: number): void {
    this.s = seed >>> 0;
  }

  /** Mulberry32 -> uniform [0, 1). */
  next(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  int(n: number): number {
    return (this.next() * n) | 0;
  }

  /** Standard normal via Box-Muller. */
  gauss(): number {
    let u = this.next();
    if (u < 1e-9) u = 1e-9;
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307179586 * v);
  }
}

/** One XorShift32 step: useful as a cheap integer hash. */
export function xorshift32(state: number): number {
  let s = state | 0;
  if (s === 0) s = 0x9e3779b9 | 0;
  s ^= s << 13;
  s ^= s >>> 17;
  s ^= s << 5;
  return s >>> 0;
}

/** Hash two integers to [0, 1). */
export function hash2(ix: number, iy: number, seed: number): number {
  const h = xorshift32((Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + seed) | 0);
  return xorshift32(h) / 4294967296;
}

const F2 = 0.5 * (Math.sqrt(3) - 1);
const G2 = (3 - Math.sqrt(3)) / 6;
const GRAD_X = new Float32Array([1, -1, 1, -1, 1, -1, 0, 0, 0.7071, -0.7071, 0.7071, -0.7071]);
const GRAD_Y = new Float32Array([1, 1, -1, -1, 0, 0, 1, -1, 0.7071, 0.7071, -0.7071, -0.7071]);

export class Simplex2D {
  private readonly perm = new Uint8Array(512);

  constructor(seed: number) {
    const rng = new Rng(seed);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
      const j = rng.int(i + 1);
      const t = p[i];
      p[i] = p[j];
      p[j] = t;
    }
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
  }

  /** Simplex noise in roughly [-1, 1]. */
  noise(xin: number, yin: number): number {
    const perm = this.perm;
    const s = (xin + yin) * F2;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const t = (i + j) * G2;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);
    const i1 = x0 > y0 ? 1 : 0;
    const j1 = x0 > y0 ? 0 : 1;
    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;
    const ii = i & 255;
    const jj = j & 255;
    let n = 0;
    let a = 0.5 - x0 * x0 - y0 * y0;
    if (a > 0) {
      const g = perm[ii + perm[jj]] % 12;
      a *= a;
      n += a * a * (GRAD_X[g] * x0 + GRAD_Y[g] * y0);
    }
    a = 0.5 - x1 * x1 - y1 * y1;
    if (a > 0) {
      const g = perm[ii + i1 + perm[jj + j1]] % 12;
      a *= a;
      n += a * a * (GRAD_X[g] * x1 + GRAD_Y[g] * y1);
    }
    a = 0.5 - x2 * x2 - y2 * y2;
    if (a > 0) {
      const g = perm[ii + 1 + perm[jj + 1]] % 12;
      a *= a;
      n += a * a * (GRAD_X[g] * x2 + GRAD_Y[g] * y2);
    }
    return 70 * n;
  }

  /** Fractal Brownian motion, normalised to roughly [-1, 1]. */
  fbm(x: number, y: number, octaves: number): number {
    let amp = 0.5;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * this.noise(x * freq, y * freq);
      norm += amp;
      amp *= 0.5;
      freq *= 2.03;
    }
    return sum / norm;
  }
}
