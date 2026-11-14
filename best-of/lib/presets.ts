/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2/lib/presets.ts)
 * Why best: build(sim, rect, cohort, rng, noise) روی هر مستطیل arbitrary، split-arena ۵۰/۵۰ با layouts identical و دیوار میانی، Open Sanctuary / Double Bridge / Labyrinth (DFS + braiding)
 * Borrow from other zips: prismic: فرمت WallStroke (circle/capsule) برای presetهای تعریفی
 * Notes: براي ادجام به best-of/lib/presets.ts
 * Status: verbatim copy — adapt import paths when merging into app/
 */

/**
 * World presets. Each builder works inside an arbitrary rectangle so the same layout can be stamped
 * once (single colony) or twice (50/50 side-by-side arena, one isolated half per cohort).
 */
import type { ColonySim } from './ant-soa-engine';
import { Rng, Simplex2D } from './prng-noise';
import type { PresetId, Rect } from './types';

export interface PresetDef {
  id: PresetId;
  title: string;
  blurb: string;
  build: (sim: ColonySim, r: Rect, cohort: number, rng: Rng, noise: Simplex2D) => void;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Organic wall ribbon: overlapping discs with noise-modulated radius (never boxy). */
function wallLine(
  sim: ColonySim, noise: Simplex2D,
  x0: number, y0: number, x1: number, y1: number, r: number,
): void {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.sqrt(dx * dx + dy * dy);
  const n = Math.max(1, Math.ceil(len / (r * 0.6)));
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    const px = x0 + dx * t;
    const py = y0 + dy * t;
    const rr = r * (1 + 0.28 * noise.noise(px * 0.021, py * 0.021));
    sim.paintWall(px, py, rr, true);
  }
}

function rock(sim: ColonySim, noise: Simplex2D, x: number, y: number, r: number): void {
  sim.paintWall(x, y, r * 0.7, true);
  for (let k = 0; k < 9; k++) {
    const a = (k / 9) * Math.PI * 2;
    const m = 0.55 + 0.45 * (0.5 + 0.5 * noise.noise(x * 0.03 + Math.cos(a), y * 0.03 + Math.sin(a)));
    sim.paintWall(x + Math.cos(a) * r * 0.5 * m, y + Math.sin(a) * r * 0.5 * m, r * 0.55 * m, true);
  }
}

function farFromFood(sim: ColonySim, x: number, y: number, d: number): boolean {
  const f = sim.food;
  for (let i = 0; i < sim.foodCount; i++) {
    if (f.active[i] === 0) continue;
    const dx = f.x[i] - x;
    const dy = f.y[i] - y;
    if (dx * dx + dy * dy < d * d) return false;
  }
  return true;
}

/* ------------------------------ Open Sanctuary ------------------------------ */

function buildOpen(sim: ColonySim, r: Rect, cohort: number, rng: Rng, noise: Simplex2D): void {
  const u = Math.min(r.w, r.h);
  const nx = r.x + r.w * (0.5 + (rng.next() - 0.5) * 0.08);
  const ny = r.y + r.h * (0.5 + (rng.next() - 0.5) * 0.08);
  sim.setNest(cohort, nx, ny);
  const n = 3 + rng.int(2);
  const base = rng.next() * Math.PI * 2;
  for (let k = 0; k < n; k++) {
    const a = base + (k / n) * Math.PI * 2 + (rng.next() - 0.5) * 0.7;
    const d = u * (0.3 + 0.12 * rng.next());
    const fx = clamp(nx + Math.cos(a) * d, r.x + 28, r.x + r.w - 28);
    const fy = clamp(ny + Math.sin(a) * d, r.y + 28, r.y + r.h - 28);
    sim.addFood(fx, fy, 130 + rng.int(70));
  }
  // elongated regions get two distant caches so the long axis is worth exploring
  const long = Math.max(r.w, r.h);
  if (long > u * 1.45) {
    const vertical = r.h > r.w;
    for (let s = -1; s <= 1; s += 2) {
      const off = long * 0.36 * s;
      const fx = clamp(nx + (vertical ? (rng.next() - 0.5) * u * 0.5 : off), r.x + 28, r.x + r.w - 28);
      const fy = clamp(ny + (vertical ? off : (rng.next() - 0.5) * u * 0.5), r.y + 28, r.y + r.h - 28);
      sim.addFood(fx, fy, 220);
    }
  }
  const rocks = rng.int(6) + (long > u * 1.45 ? 3 : 0);
  for (let k = 0; k < rocks; k++) {
    for (let tries = 0; tries < 12; tries++) {
      const px = r.x + 30 + rng.next() * (r.w - 60);
      const py = r.y + 30 + rng.next() * (r.h - 60);
      const dx = px - nx;
      const dy = py - ny;
      if (dx * dx + dy * dy < 80 * 80) continue;
      if (!farFromFood(sim, px, py, 52)) continue;
      rock(sim, noise, px, py, u * (0.035 + 0.035 * rng.next()));
      break;
    }
  }
}

/* -------------------------- Deneubourg's Double Bridge -------------------------- */

function buildBridge(sim: ColonySim, r: Rect, cohort: number, rng: Rng, noise: Simplex2D): void {
  const vertical = r.h >= r.w;
  const L = vertical ? r.h : r.w;
  const C = vertical ? r.w : r.h;
  // (t along the long axis, s across) -> world
  const wx = (t: number, s: number): number => (vertical ? r.x + s * C : r.x + t * L);
  const wy = (t: number, s: number): number => (vertical ? r.y + t * L : r.y + s * C);
  const flip = rng.next() < 0.5;
  const S = (s: number): number => (flip ? 1 - s : s);

  sim.setNest(cohort, wx(0.15, S(0.42)), wy(0.15, S(0.42)));
  sim.addFood(wx(0.86, S(0.3)), wy(0.86, S(0.3)), 220);
  sim.addFood(wx(0.88, S(0.62)), wy(0.88, S(0.62)), 220);

  const gapHalf = Math.max(26, C * 0.07) / C;
  const shortS = 0.42;
  const longS = 0.9;
  const t0 = 0.42;
  const t1 = 0.58;
  const rr = 9;
  const stepS = (rr * 1.1) / C;
  const stepT = (rr * 1.1) / L;
  for (let t = t0; t <= t1; t += stepT) {
    for (let s = 0; s <= 1.0001; s += stepS) {
      if (Math.abs(s - shortS) < gapHalf) continue; // short branch
      if (Math.abs(s - longS) < gapHalf) continue; // long branch
      const px = wx(t, S(s));
      const py = wy(t, S(s));
      sim.paintWall(px, py, rr * (1 + 0.25 * noise.noise(px * 0.02, py * 0.02)), true);
    }
  }
  // baffles lengthen the far branch on both banks
  wallLine(sim, noise, wx(0.31, S(0.74)), wy(0.31, S(0.74)), wx(0.31, S(1)), wy(0.31, S(1)), 7);
  wallLine(sim, noise, wx(0.69, S(0.74)), wy(0.69, S(0.74)), wx(0.69, S(1)), wy(0.69, S(1)), 7);
}

/* ---------------------------- Subterranean Labyrinth ---------------------------- */

function buildLabyrinth(sim: ColonySim, r: Rect, cohort: number, rng: Rng, noise: Simplex2D): void {
  const u = Math.min(r.w, r.h);
  const cell = clamp(u / 4.2, 84, 150);
  const mc = Math.max(2, Math.floor(r.w / cell));
  const mr = Math.max(2, Math.floor(r.h / cell));
  const cw = r.w / mc;
  const chh = r.h / mr;
  const total = mc * mr;
  // wall flags: east wall of (x,y) and south wall of (x,y)
  const east = new Uint8Array(total).fill(1);
  const south = new Uint8Array(total).fill(1);
  const seen = new Uint8Array(total);
  const stack = new Int32Array(total);
  let top = 0;
  const start = ((mr >> 1) * mc + (mc >> 1)) | 0;
  stack[top++] = start;
  seen[start] = 1;
  while (top > 0) {
    const cur = stack[top - 1];
    const cx = cur % mc;
    const cy = (cur / mc) | 0;
    let opts = 0;
    let pick = -1;
    for (let d = 0; d < 4; d++) {
      const nx = cx + (d === 0 ? 1 : d === 1 ? -1 : 0);
      const ny = cy + (d === 2 ? 1 : d === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= mc || ny >= mr || seen[ny * mc + nx] !== 0) continue;
      opts++;
      if (rng.int(opts) === 0) pick = d;
    }
    if (pick < 0) {
      top--;
      continue;
    }
    const nx = cx + (pick === 0 ? 1 : pick === 1 ? -1 : 0);
    const ny = cy + (pick === 2 ? 1 : pick === 3 ? -1 : 0);
    if (pick === 0) east[cur] = 0;
    else if (pick === 1) east[ny * mc + nx] = 0;
    else if (pick === 2) south[cur] = 0;
    else south[ny * mc + nx] = 0;
    seen[ny * mc + nx] = 1;
    stack[top++] = ny * mc + nx;
  }
  // braid: knock out extra walls so there are loops and alternative routes
  for (let i = 0; i < total; i++) {
    if (rng.next() < 0.3) east[i] = 0;
    if (rng.next() < 0.3) south[i] = 0;
  }
  const wr = 6.5;
  for (let cy = 0; cy < mr; cy++) {
    for (let cx = 0; cx < mc; cx++) {
      const i = cy * mc + cx;
      const x0 = r.x + cx * cw;
      const y0 = r.y + cy * chh;
      if (cx < mc - 1 && east[i] !== 0) {
        wallLine(sim, noise, x0 + cw, y0 + chh * 0.08, x0 + cw, y0 + chh * 0.92, wr);
      }
      if (cy < mr - 1 && south[i] !== 0) {
        wallLine(sim, noise, x0 + cw * 0.08, y0 + chh, x0 + cw * 0.92, y0 + chh, wr);
      }
    }
  }
  const sx = start % mc;
  const sy = (start / mc) | 0;
  sim.setNest(cohort, r.x + (sx + 0.5) * cw, r.y + (sy + 0.5) * chh);
  const caches = Math.min(total - 1, 4 + rng.int(2));
  const used = new Uint8Array(total);
  used[start] = 1;
  for (let k = 0; k < caches; k++) {
    for (let tries = 0; tries < 20; tries++) {
      const ci = rng.int(total);
      if (used[ci] !== 0) continue;
      used[ci] = 1;
      sim.addFood(r.x + ((ci % mc) + 0.5) * cw, r.y + (((ci / mc) | 0) + 0.5) * chh, 150);
      break;
    }
  }
}

export const PRESETS: readonly PresetDef[] = [
  {
    id: 'open',
    title: 'Open Sanctuary',
    blurb: 'Scattered emerald caches around a central brood reactor. Pure trail formation.',
    build: buildOpen,
  },
  {
    id: 'bridge',
    title: "Deneubourg's Double Bridge",
    blurb: 'A basalt barrier with a short and a long branch. Watch the colony pick one.',
    build: buildBridge,
  },
  {
    id: 'labyrinth',
    title: 'Subterranean Labyrinth',
    blurb: 'A braided maze. Dead ends get sealed with viridian noEntry.',
    build: buildLabyrinth,
  },
];

export function getPreset(id: PresetId): PresetDef {
  for (const p of PRESETS) if (p.id === id) return p;
  return PRESETS[0];
}

/** Clears the world, stamps the preset (twice with a divider in A/B arenas) and spawns the population. */
export function applyPreset(sim: ColonySim, id: PresetId, seed: number, population: number): void {
  const def = getPreset(id);
  sim.clearWorld();
  const noise = new Simplex2D(seed + 17);
  const split = sim.arena === 'ab-random' || sim.arena === 'ab-aco';
  if (!split) {
    sim.nestCount = 1;
    def.build(sim, { x: 0, y: 0, w: sim.width, h: sim.height }, 0, new Rng(seed), noise);
  } else {
    sim.nestCount = 2;
    const landscape = sim.width >= sim.height;
    const g = 8;
    const ra: Rect = landscape
      ? { x: 0, y: 0, w: sim.width / 2 - g, h: sim.height }
      : { x: 0, y: 0, w: sim.width, h: sim.height / 2 - g };
    const rb: Rect = landscape
      ? { x: sim.width / 2 + g, y: 0, w: sim.width / 2 - g, h: sim.height }
      : { x: 0, y: sim.height / 2 + g, w: sim.width, h: sim.height / 2 - g };
    // identical layouts (same seed) translated into each half
    def.build(sim, ra, 0, new Rng(seed), noise);
    def.build(sim, rb, 1, new Rng(seed), noise);
    if (landscape) {
      for (let y = 0; y <= sim.height; y += 5) sim.grid.setWallDisc(sim.width / 2, y, 7.5, true);
    } else {
      for (let x = 0; x <= sim.width; x += 5) sim.grid.setWallDisc(x, sim.height / 2, 7.5, true);
    }
  }
  sim.placeGraveyards();
  sim.populate(Math.min(population, sim.params.maxPop));
}
