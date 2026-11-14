/**
 * Procedural art (browser only): veined abyssal soil, terraced basalt with bioluminescent rims,
 * articulated ant sprite sheets, armoured predator, emerald crystals, chitin husk, vignette + grain.
 * Everything is drawn with vector canvas ops or ImageData. No blur filters anywhere.
 */
import type { PheromoneGrid } from './pheromone-grid';
import { PHEROMONE_CELL } from './pheromone-grid';
import { Rng, Simplex2D, hash2 } from './prng-noise';

export const SOIL_HEX = '#020307';
export const WALL_HEX = '#0d1322';
export const RIM_TEAL = '#2dd4bf';
export const RIM_SKY = '#38bdf8';
export const CRYSTAL_HEX = '#00e676';
export const FACET_HEX = '#a7f3d0';

export const ANT_CELL = 28;
export const ANT_FRAMES = 8;
export const PREDATOR_CELL = 64;
export const CRYSTAL_CELL = 48;
export const CRYSTAL_VARIANTS = 8;
export const HUSK_CELL = 24;

export function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.floor(w));
  c.height = Math.max(1, Math.floor(h));
  return c;
}

export function ctx2d(c: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = c.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  return ctx;
}

/* ------------------------------------------------------------------ */
/* Ant sprites                                                         */
/* ------------------------------------------------------------------ */

export interface AntPalette {
  body: string;
  shade: string;
  leg: string;
  glowA: string;
  glowB: string;
}

/** 4 caste rows: SCOUT violet-cyan, HAULER amber-gold, WARDEN viridian-crimson, DRONE rose-slate. */
export const CASTE_PALETTES: readonly AntPalette[] = [
  { body: '#8b7cf6', shade: '#3b2f86', leg: '#a5b4fc', glowA: '#c77dff', glowB: '#00e1ff' },
  { body: '#d9962b', shade: '#6b430c', leg: '#f2c779', glowA: '#ffa800', glowB: '#ffe28a' },
  { body: '#3fbf6a', shade: '#125a2c', leg: '#9be7b0', glowA: '#50ff64', glowB: '#ff0c4b' },
  { body: '#b68a98', shade: '#4d3a48', leg: '#cbd5e1', glowA: '#fb7185', glowB: '#94a3b8' },
];

/** A/B cohort rows: TRAINED gold-cyan, RIVAL (Random / ACO) rose-slate. */
export const COHORT_PALETTES: readonly AntPalette[] = [
  { body: '#e6c34a', shade: '#6e5a12', leg: '#fde68a', glowA: '#ffd24a', glowB: '#00e1ff' },
  { body: '#b0637a', shade: '#47303c', leg: '#cbd5e1', glowA: '#fb7185', glowB: '#94a3b8' },
];

function drawAnt(ctx: CanvasRenderingContext2D, cx: number, cy: number, frame: number, pal: AntPalette, laden: boolean): void {
  ctx.save();
  ctx.translate(cx, cy);
  const ph = (frame / ANT_FRAMES) * Math.PI * 2;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = pal.leg;
  ctx.lineWidth = 1.05;
  // alternating tripod gait: three legs per side
  for (let side = -1; side <= 1; side += 2) {
    for (let k = 0; k < 3; k++) {
      const sw = Math.sin(ph + ((k + (side > 0 ? 1 : 0)) % 2) * Math.PI) * 2.3;
      const ax = -2.2 + k * 2.6;
      ctx.beginPath();
      ctx.moveTo(ax, side * 1.4);
      ctx.lineTo(ax + sw * 0.45 + (k - 1) * 1.4, side * 5.2);
      ctx.lineTo(ax + sw + (k - 1) * 2.6, side * 8.6);
      ctx.stroke();
    }
    // antenna
    const wig = Math.sin(ph * 2 + side) * 0.7;
    ctx.beginPath();
    ctx.moveTo(6.2, side * 1.1);
    ctx.quadraticCurveTo(9, side * 2.2, 11.2, side * (4 + wig));
    ctx.stroke();
  }
  // abdomen, thorax, head
  ctx.fillStyle = pal.shade;
  ctx.beginPath();
  ctx.ellipse(-6.4, 0, 4.9, 3.5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = pal.body;
  ctx.beginPath();
  ctx.ellipse(-6.1, -0.3, 4.1, 2.7, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(-0.4, 0, 2.9, 2.1, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(4.8, 0, 2.7, 2.4, 0, 0, Math.PI * 2);
  ctx.fill();
  // bioluminescent organs
  ctx.fillStyle = pal.glowA;
  ctx.beginPath();
  ctx.ellipse(-6.8, 0, 2.3, 1.3, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.arc(-7.4, -0.4, 0.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = pal.glowB;
  ctx.beginPath();
  ctx.arc(5.6, -1.1, 0.8, 0, Math.PI * 2);
  ctx.arc(5.6, 1.1, 0.8, 0, Math.PI * 2);
  ctx.fill();
  if (laden) {
    ctx.fillStyle = CRYSTAL_HEX;
    ctx.beginPath();
    ctx.moveTo(8.2, 0);
    ctx.lineTo(10.6, -2.4);
    ctx.lineTo(13, 0);
    ctx.lineTo(10.6, 2.4);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = FACET_HEX;
    ctx.beginPath();
    ctx.moveTo(8.2, 0);
    ctx.lineTo(10.6, -2.4);
    ctx.lineTo(10.6, 0);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

/** Sheet layout: row = palette * 2 + (laden ? 1 : 0), column = gait frame (8). */
export function buildAntSheet(palettes: readonly AntPalette[]): HTMLCanvasElement {
  const c = makeCanvas(ANT_CELL * ANT_FRAMES, ANT_CELL * palettes.length * 2);
  const ctx = ctx2d(c);
  for (let p = 0; p < palettes.length; p++) {
    for (let laden = 0; laden < 2; laden++) {
      for (let f = 0; f < ANT_FRAMES; f++) {
        drawAnt(ctx, f * ANT_CELL + ANT_CELL / 2 - 1, (p * 2 + laden) * ANT_CELL + ANT_CELL / 2, f, palettes[p], laden === 1);
      }
    }
  }
  return c;
}

/* ------------------------------------------------------------------ */
/* Abyssal Stalker                                                     */
/* ------------------------------------------------------------------ */

export function buildPredatorSheet(): HTMLCanvasElement {
  const c = makeCanvas(PREDATOR_CELL * ANT_FRAMES, PREDATOR_CELL);
  const ctx = ctx2d(c);
  for (let f = 0; f < ANT_FRAMES; f++) {
    ctx.save();
    ctx.translate(f * PREDATOR_CELL + PREDATOR_CELL / 2, PREDATOR_CELL / 2);
    const ph = (f / ANT_FRAMES) * Math.PI * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    // eight legs
    ctx.strokeStyle = '#59647f';
    ctx.lineWidth = 2;
    for (let side = -1; side <= 1; side += 2) {
      for (let k = 0; k < 4; k++) {
        const sw = Math.sin(ph + ((k + (side > 0 ? 1 : 0)) % 2) * Math.PI) * 4;
        const ax = -12 + k * 7;
        ctx.beginPath();
        ctx.moveTo(ax, side * 6);
        ctx.lineTo(ax + sw * 0.5 + (k - 1.5) * 2, side * 16);
        ctx.lineTo(ax + sw + (k - 1.5) * 4.5, side * 25);
        ctx.stroke();
      }
      // pincers
      ctx.strokeStyle = '#8b96b4';
      ctx.lineWidth = 2.4;
      const open = 3 + Math.sin(ph) * 2;
      ctx.beginPath();
      ctx.moveTo(17, side * 3);
      ctx.quadraticCurveTo(26, side * (6 + open), 28, side * (1 + open * 0.4));
      ctx.stroke();
      ctx.strokeStyle = '#59647f';
      ctx.lineWidth = 2;
    }
    // armoured plates, tail to head
    const plates: Array<[number, number, number]> = [[-19, 5.5, 5], [-12.5, 8, 7], [-4, 9.5, 8.5], [5, 8.5, 7.6], [13.5, 7, 6.4]];
    for (const [px, rx, ry] of plates) {
      ctx.fillStyle = '#0a0e18';
      ctx.beginPath();
      ctx.ellipse(px, 0, rx + 1.2, ry + 1.2, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#1b2233';
      ctx.beginPath();
      ctx.ellipse(px, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#3d4863';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.ellipse(px, 0, rx - 1.6, ry - 1.6, 0, -1.2, 1.2);
      ctx.stroke();
    }
    // dorsal ridge
    ctx.strokeStyle = '#ff0c4b';
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(-20, 0);
    ctx.lineTo(8, 0);
    ctx.stroke();
    ctx.globalAlpha = 1;
    // crimson oculars
    for (let side = -1; side <= 1; side += 2) {
      ctx.fillStyle = '#ff0c4b';
      ctx.beginPath();
      ctx.arc(16, side * 3.2, 2.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffd5df';
      ctx.beginPath();
      ctx.arc(16.6, side * 3.2, 0.9, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  return c;
}

/* ------------------------------------------------------------------ */
/* Emerald crystals + husk                                             */
/* ------------------------------------------------------------------ */

export function buildCrystalSheet(seed: number): HTMLCanvasElement {
  const c = makeCanvas(CRYSTAL_CELL * CRYSTAL_VARIANTS, CRYSTAL_CELL);
  const ctx = ctx2d(c);
  const rng = new Rng(seed);
  for (let v = 0; v < CRYSTAL_VARIANTS; v++) {
    ctx.save();
    ctx.translate(v * CRYSTAL_CELL + CRYSTAL_CELL / 2, CRYSTAL_CELL / 2);
    const shards = 4 + rng.int(3);
    const base = rng.next() * Math.PI * 2;
    for (let s = 0; s < shards; s++) {
      const a = base + (s / shards) * Math.PI * 2 + (rng.next() - 0.5) * 0.5;
      const len = 11 + rng.next() * 10;
      const wdt = 4 + rng.next() * 3.5;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      const tx = ca * len;
      const ty = sa * len;
      const mx = ca * len * 0.45;
      const my = sa * len * 0.45;
      const nx = -sa * wdt;
      const ny = ca * wdt;
      // dark flank
      ctx.fillStyle = '#00994d';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(mx - nx, my - ny);
      ctx.lineTo(tx, ty);
      ctx.closePath();
      ctx.fill();
      // body
      ctx.fillStyle = CRYSTAL_HEX;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(mx + nx, my + ny);
      ctx.lineTo(tx, ty);
      ctx.closePath();
      ctx.fill();
      // lit facet
      ctx.fillStyle = FACET_HEX;
      ctx.beginPath();
      ctx.moveTo(mx + nx * 0.95, my + ny * 0.95);
      ctx.lineTo(mx + nx * 0.2, my + ny * 0.2);
      ctx.lineTo(tx, ty);
      ctx.closePath();
      ctx.fill();
    }
    // core gem
    ctx.fillStyle = '#5dffa8';
    ctx.beginPath();
    for (let k = 0; k < 6; k++) {
      const a = base + (k / 6) * Math.PI * 2;
      const r = 5.5 + (k % 2) * 1.5;
      if (k === 0) ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r);
      else ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fill();
    // sparks
    ctx.fillStyle = '#eafff5';
    for (let k = 0; k < 3; k++) {
      const a = rng.next() * Math.PI * 2;
      const d = 6 + rng.next() * 12;
      const sx = Math.cos(a) * d;
      const sy = Math.sin(a) * d;
      ctx.beginPath();
      ctx.moveTo(sx - 2.2, sy);
      ctx.lineTo(sx, sy - 0.6);
      ctx.lineTo(sx + 2.2, sy);
      ctx.lineTo(sx, sy + 0.6);
      ctx.closePath();
      ctx.moveTo(sx, sy - 2.2);
      ctx.lineTo(sx + 0.6, sy);
      ctx.lineTo(sx, sy + 2.2);
      ctx.lineTo(sx - 0.6, sy);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }
  return c;
}

export function buildHusk(): HTMLCanvasElement {
  const c = makeCanvas(HUSK_CELL, HUSK_CELL);
  const ctx = ctx2d(c);
  ctx.translate(HUSK_CELL / 2, HUSK_CELL / 2);
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#7a6c5e';
  ctx.lineWidth = 1;
  for (let side = -1; side <= 1; side += 2) {
    for (let k = 0; k < 3; k++) {
      const ax = -2 + k * 2.4;
      ctx.beginPath();
      ctx.moveTo(ax, side * 1.2);
      ctx.quadraticCurveTo(ax + 2.5, side * 5.5, ax - 0.5, side * 4.2); // curled in death
      ctx.stroke();
    }
  }
  ctx.fillStyle = '#3d342c';
  ctx.beginPath();
  ctx.ellipse(-5, 0.6, 4.2, 3, 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#5b4e42';
  ctx.beginPath();
  ctx.ellipse(-4.8, 0.2, 3.3, 2.2, 0.3, 0, Math.PI * 2);
  ctx.ellipse(0.2, 0, 2.4, 1.8, 0, 0, Math.PI * 2);
  ctx.ellipse(4, -0.6, 2.2, 2, -0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#8f8172';
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.arc(-4.8, 0.2, 2.2, -2.4, -0.6);
  ctx.stroke();
  return c;
}

/* ------------------------------------------------------------------ */
/* Terrain: veined soil + terraced basalt                              */
/* ------------------------------------------------------------------ */

/**
 * Repaints a world-space rectangle of the static terrain cache (1 terrain px = 1 world px).
 * Callers pass a dirty rect already expanded by the smoothing margin.
 */
export function paintTerrainRegion(
  ctx: CanvasRenderingContext2D,
  grid: PheromoneGrid,
  noise: Simplex2D,
  seed: number,
  rx: number,
  ry: number,
  rw: number,
  rh: number,
): void {
  if (rw <= 0 || rh <= 0) return;
  // --- soil #020307 with crystalline micro-veins (never brighter than #090e1a) ---
  const img = ctx.createImageData(rw, rh);
  const d = img.data;
  for (let py = 0; py < rh; py++) {
    const wy = ry + py;
    for (let px = 0; px < rw; px++) {
      const wx = rx + px;
      const n1 = noise.noise(wx * 0.0125, wy * 0.0125);
      const n2 = noise.noise(wx * 0.041 + 31.7, wy * 0.041 - 12.3);
      let vein = 1 - Math.abs(n1 + n2 * 0.35);
      vein = vein < 0 ? 0 : vein;
      vein = vein * vein;
      vein = vein * vein;
      vein = vein * vein; // sharp ridges
      const grain = hash2(wx, wy, seed) * 1.2;
      const o = (py * rw + px) * 4;
      d[o] = 2 + vein * 7 * 0.95 + grain * 0.4;
      d[o + 1] = 3 + vein * 11 * 0.95 + grain * 0.5;
      d[o + 2] = 7 + vein * 19 * 0.95 + grain;
      d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, rx, ry);

  ctx.save();
  ctx.beginPath();
  ctx.rect(rx, ry, rw, rh);
  ctx.clip();

  // --- survey reticles ---
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.07)';
  ctx.lineWidth = 1;
  const step = 120;
  const gx0 = Math.floor(rx / step) * step;
  const gy0 = Math.floor(ry / step) * step;
  ctx.beginPath();
  for (let gy = gy0; gy <= ry + rh + step; gy += step) {
    for (let gx = gx0; gx <= rx + rw + step; gx += step) {
      ctx.moveTo(gx - 5, gy + 0.5);
      ctx.lineTo(gx + 5, gy + 0.5);
      ctx.moveTo(gx + 0.5, gy - 5);
      ctx.lineTo(gx + 0.5, gy + 5);
    }
  }
  ctx.stroke();

  // --- walls: AO halo, bioluminescent rim, basalt body, two terraces ---
  const cs = PHEROMONE_CELL;
  const cols = grid.cols;
  const rows = grid.rows;
  const wall = grid.wall;
  const cx0 = Math.max(0, Math.floor(rx / cs) - 3);
  const cy0 = Math.max(0, Math.floor(ry / cs) - 3);
  const cx1 = Math.min(cols - 1, Math.floor((rx + rw) / cs) + 3);
  const cy1 = Math.min(rows - 1, Math.floor((ry + rh) / cs) + 3);
  const TAU = Math.PI * 2;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.22)';
  for (let gy = cy0; gy <= cy1; gy++) {
    for (let gx = cx0; gx <= cx1; gx++) {
      if (wall[gy * cols + gx] === 0) continue;
      ctx.beginPath();
      ctx.arc((gx + 0.5) * cs, (gy + 0.5) * cs + 1.5, cs * 1.7, 0, TAU);
      ctx.fill();
    }
  }
  for (let gy = cy0; gy <= cy1; gy++) {
    for (let gx = cx0; gx <= cx1; gx++) {
      if (wall[gy * cols + gx] === 0) continue;
      ctx.fillStyle = noise.noise(gx * 0.11, gy * 0.11) > 0 ? RIM_TEAL : RIM_SKY;
      ctx.beginPath();
      ctx.arc((gx + 0.5) * cs, (gy + 0.5) * cs, cs * 1.02, 0, TAU);
      ctx.fill();
    }
  }
  ctx.fillStyle = WALL_HEX;
  for (let gy = cy0; gy <= cy1; gy++) {
    for (let gx = cx0; gx <= cx1; gx++) {
      if (wall[gy * cols + gx] === 0) continue;
      ctx.beginPath();
      ctx.arc((gx + 0.5) * cs, (gy + 0.5) * cs, cs * 1.02 - 1.3, 0, TAU);
      ctx.fill();
    }
  }
  // terrace 1: cells whose 4-neighbourhood is solid; terrace 2: 5x5 solid
  for (let pass = 1; pass <= 2; pass++) {
    ctx.fillStyle = pass === 1 ? '#111a2d' : '#17213a';
    for (let gy = cy0; gy <= cy1; gy++) {
      for (let gx = cx0; gx <= cx1; gx++) {
        const i = gy * cols + gx;
        if (wall[i] === 0) continue;
        let solid = true;
        for (let oy = -pass; oy <= pass && solid; oy++) {
          for (let ox = -pass; ox <= pass; ox++) {
            const nx = gx + ox;
            const ny = gy + oy;
            if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
            if (wall[ny * cols + nx] === 0) {
              solid = false;
              break;
            }
          }
        }
        if (!solid) continue;
        ctx.beginPath();
        ctx.arc((gx + 0.5) * cs - pass * 0.6, (gy + 0.5) * cs - pass * 0.8, cs * (pass === 1 ? 0.84 : 0.78), 0, TAU);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

/* ------------------------------------------------------------------ */
/* Vignette + grain overlay                                            */
/* ------------------------------------------------------------------ */

export function buildOverlay(w: number, h: number, seed: number): HTMLCanvasElement {
  const c = makeCanvas(w, h);
  const ctx = ctx2d(c);
  const g = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.32, w / 2, h / 2, Math.hypot(w, h) * 0.56);
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(0.7, 'rgba(0,0,0,0.22)');
  g.addColorStop(1, 'rgba(0,0,0,0.62)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  // film grain tile
  const tile = makeCanvas(96, 96);
  const tctx = ctx2d(tile);
  const img = tctx.createImageData(96, 96);
  const rng = new Rng(seed);
  for (let i = 0; i < 96 * 96; i++) {
    const v = rng.next();
    const o = i * 4;
    const bright = v > 0.5;
    img.data[o] = bright ? 255 : 0;
    img.data[o + 1] = bright ? 255 : 0;
    img.data[o + 2] = bright ? 255 : 0;
    img.data[o + 3] = Math.abs(v - 0.5) * 14;
  }
  tctx.putImageData(img, 0, 0);
  const pat = ctx.createPattern(tile, 'repeat');
  if (pat) {
    ctx.fillStyle = pat;
    ctx.fillRect(0, 0, w, h);
  }
  return c;
}
