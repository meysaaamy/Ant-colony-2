/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2/lib/renderer.ts)
 * Why best: رندرر ۱۰ لایه بدون blur/shadowBlur: کش terrain + dirty-rect، تصویر فيرمون 'screen'، bloom ۱/۴ رزولوشن، adaptive scaler (زیر ۵۰ FPS difusión/ bloom کاهش، فيزیک ۱۰۰٪)
 * Borrow from other zips: هیچ (بهترین رندرر)
 * Notes: براي ادجام به best-of/lib/renderer.ts
 * Status: verbatim copy — adapt import paths when merging into app/
 */

/**
 * Strict zero-blur 10-layer compositor (no ctx.filter, no shadowBlur anywhere):
 *  1 static terrain cache (dirty-rect repaint + smoothing margin)
 *  2 full-spectrum 5-channel RGB pheromone ImageData, composited with 'screen'
 *  3 1/4-res prismatic bloom, refreshed every 2nd frame, bilinear upscaled with 'lighter'
 *  4 chitin corpse husks + emerald crystals
 *  5 brood reactor nest crater (harmonic mandala, stored food arc, larval pods, GEN aura)
 *  6 articulated ants (batched rotated drawImage) + Abyssal Stalkers
 *  7 synaptic / combat arcs
 *  8 atmospheric spore motes
 *  9 active brush ribbon
 * 10 cached vignette + grain
 * Adaptive scaler: below 50 FPS (60-frame window) diffusion drops to every 2nd tick and bloom is shed;
 * ants and physics always stay at 100%.
 */
import { ColonySim, MAX_ARCS, MAX_PREDATORS, NEST_RADIUS, FOOD_PER_BIRTH, BIRTHS_PER_GENERATION } from './ant-soa-engine';
import { DEFAULT_GAMMA, PHEROMONE_CELL } from './pheromone-grid';
import {
  ANT_CELL, CASTE_PALETTES, COHORT_PALETTES, CRYSTAL_CELL, HUSK_CELL, PREDATOR_CELL,
  buildAntSheet, buildCrystalSheet, buildHusk, buildOverlay, buildPredatorSheet, ctx2d, makeCanvas, paintTerrainRegion,
} from './procedural-assets';
import { Rng, Simplex2D } from './prng-noise';
import { ARC_COMBAT, ARC_TANDEM, POLICY_ACO, STATE_DEAD } from './types';

export const MAX_MOTES = 45;
export const TRAIL_POINTS = 48;
const TERRAIN_MARGIN = 14;
const DASH: number[] = [7, 6];
const NO_DASH: number[] = [];
const TAU = Math.PI * 2;

export interface BrushVisual {
  visible: boolean;
  down: boolean;
  x: number;
  y: number;
  radius: number;
  erase: boolean;
  color: string;
  trail: Float32Array;
  trailCount: number;
  startX: number;
  startY: number;
  startTime: number;
}

export function createBrushVisual(): BrushVisual {
  return {
    visible: false, down: false, x: 0, y: 0, radius: 24, erase: false, color: 'rgb(255,168,0)',
    trail: new Float32Array(TRAIL_POINTS * 2), trailCount: 0, startX: 0, startY: 0, startTime: -1e9,
  };
}

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly sim: ColonySim;
  private readonly noise: Simplex2D;
  private readonly seed: number;

  dpr = 1;
  cssW = 1;
  cssH = 1;
  /** world px -> css px */
  scale = 1;
  /** device px offsets of the world origin */
  offX = 0;
  offY = 0;

  private readonly terrain: HTMLCanvasElement;
  private readonly tctx: CanvasRenderingContext2D;
  private dirtyX0 = 0;
  private dirtyY0 = 0;
  private dirtyX1 = -1;
  private dirtyY1 = -1;

  private readonly pher: HTMLCanvasElement;
  private readonly pctx: CanvasRenderingContext2D;
  private readonly pherImage: ImageData;
  private readonly bloom: HTMLCanvasElement;
  private readonly bctx: CanvasRenderingContext2D;

  private readonly casteSheet: HTMLCanvasElement;
  private readonly cohortSheet: HTMLCanvasElement;
  private readonly predatorSheet: HTMLCanvasElement;
  private readonly crystalSheet: HTMLCanvasElement;
  private readonly husk: HTMLCanvasElement;
  private overlay: HTMLCanvasElement | null = null;

  private readonly moteX = new Float32Array(MAX_MOTES);
  private readonly moteY = new Float32Array(MAX_MOTES);
  private readonly moteZ = new Float32Array(MAX_MOTES);
  private readonly motePhase = new Float32Array(MAX_MOTES);

  gamma = DEFAULT_GAMMA;
  adaptive = false;
  fps = 60;
  private frame = 0;
  private fpsFrames = 0;
  private fpsTime = 0;
  private goodWindows = 0;
  private lastNow = 0;
  private fontFamily = 'system-ui, sans-serif';
  private genCache = -1;
  private genLabel = 'GEN 1';

  constructor(canvas: HTMLCanvasElement, sim: ColonySim, seed: number) {
    this.canvas = canvas;
    this.ctx = ctx2d(canvas);
    this.sim = sim;
    this.seed = seed;
    this.noise = new Simplex2D(seed);
    this.terrain = makeCanvas(sim.width, sim.height);
    this.tctx = ctx2d(this.terrain);
    this.pher = makeCanvas(sim.grid.cols, sim.grid.rows);
    this.pctx = ctx2d(this.pher);
    this.pherImage = this.pctx.createImageData(sim.grid.cols, sim.grid.rows);
    this.bloom = makeCanvas(Math.ceil(sim.grid.cols / 4), Math.ceil(sim.grid.rows / 4));
    this.bctx = ctx2d(this.bloom);
    this.casteSheet = buildAntSheet(CASTE_PALETTES);
    this.cohortSheet = buildAntSheet(COHORT_PALETTES);
    this.predatorSheet = buildPredatorSheet();
    this.crystalSheet = buildCrystalSheet(seed + 5);
    this.husk = buildHusk();
    const rng = new Rng(seed + 11);
    for (let i = 0; i < MAX_MOTES; i++) {
      this.moteX[i] = rng.next();
      this.moteY[i] = rng.next();
      this.moteZ[i] = 0.3 + rng.next() * 0.7;
      this.motePhase[i] = rng.next() * TAU;
    }
    this.invalidateTerrain();
  }

  setFont(family: string): void {
    if (family) this.fontFamily = family;
  }

  /** State-preserving resize: only the view transform and overlay change, the world is untouched. */
  resize(cssW: number, cssH: number, devicePixelRatio: number): void {
    this.dpr = Math.min(2, Math.max(1, devicePixelRatio));
    this.cssW = Math.max(1, cssW);
    this.cssH = Math.max(1, cssH);
    this.canvas.width = Math.round(this.cssW * this.dpr);
    this.canvas.height = Math.round(this.cssH * this.dpr);
    this.scale = Math.min(this.cssW / this.sim.width, this.cssH / this.sim.height);
    this.offX = ((this.cssW - this.sim.width * this.scale) / 2) * this.dpr;
    this.offY = ((this.cssH - this.sim.height * this.scale) / 2) * this.dpr;
    this.overlay = buildOverlay(Math.ceil(this.canvas.width / 2), Math.ceil(this.canvas.height / 2), this.seed + 3);
  }

  /** Client (css px, relative to the canvas box) -> world coordinates. */
  toWorld(localX: number, localY: number, out: Float32Array): void {
    out[0] = (localX - this.offX / this.dpr) / this.scale;
    out[1] = (localY - this.offY / this.dpr) / this.scale;
  }

  invalidateTerrain(): void {
    this.dirtyX0 = 0;
    this.dirtyY0 = 0;
    this.dirtyX1 = this.sim.width;
    this.dirtyY1 = this.sim.height;
  }

  /** Accumulates a dirty rect (world px) around a wall edit; margin keeps contours smooth at the seams. */
  markTerrainDirty(x: number, y: number, r: number): void {
    const x0 = x - r - TERRAIN_MARGIN;
    const y0 = y - r - TERRAIN_MARGIN;
    const x1 = x + r + TERRAIN_MARGIN;
    const y1 = y + r + TERRAIN_MARGIN;
    if (this.dirtyX1 < this.dirtyX0) {
      this.dirtyX0 = x0;
      this.dirtyY0 = y0;
      this.dirtyX1 = x1;
      this.dirtyY1 = y1;
    } else {
      if (x0 < this.dirtyX0) this.dirtyX0 = x0;
      if (y0 < this.dirtyY0) this.dirtyY0 = y0;
      if (x1 > this.dirtyX1) this.dirtyX1 = x1;
      if (y1 > this.dirtyY1) this.dirtyY1 = y1;
    }
  }

  private flushTerrain(): void {
    if (this.dirtyX1 < this.dirtyX0) return;
    const x0 = Math.max(0, Math.floor(this.dirtyX0));
    const y0 = Math.max(0, Math.floor(this.dirtyY0));
    const x1 = Math.min(this.sim.width, Math.ceil(this.dirtyX1));
    const y1 = Math.min(this.sim.height, Math.ceil(this.dirtyY1));
    paintTerrainRegion(this.tctx, this.sim.grid, this.noise, this.seed, x0, y0, x1 - x0, y1 - y0);
    this.dirtyX1 = -1;
    this.dirtyX0 = 0;
  }

  /** Feeds the adaptive scaler. Call once per rendered frame with the real frame delta. */
  noteFrame(dtMs: number): void {
    this.fpsFrames++;
    this.fpsTime += dtMs;
    if (this.fpsFrames < 60) return;
    this.fps = (1000 * this.fpsFrames) / Math.max(1, this.fpsTime);
    this.fpsFrames = 0;
    this.fpsTime = 0;
    if (this.fps < 50) {
      this.adaptive = true;
      this.goodWindows = 0;
    } else if (this.adaptive && this.fps >= 57) {
      if (++this.goodWindows >= 4) this.adaptive = false;
    }
    this.sim.halfRateDiffusion = this.adaptive;
  }

  private rivalLabel(): string {
    return this.sim.policyB === POLICY_ACO ? 'HAND-CRAFTED ACO' : 'RANDOM BRAIN';
  }

  private drawNest(c: number, now: number, ab: boolean): void {
    const ctx = this.ctx;
    const sim = this.sim;
    const nx = sim.nestX[c];
    const ny = sim.nestY[c];
    const t = now * 0.001;
    // crater floor
    ctx.fillStyle = '#04070d';
    ctx.beginPath();
    ctx.arc(nx, ny, NEST_RADIUS + 3, 0, TAU);
    ctx.fill();
    ctx.fillStyle = '#070c16';
    ctx.beginPath();
    ctx.arc(nx, ny, NEST_RADIUS - 6, 0, TAU);
    ctx.fill();
    // harmonic mandala: three counter-rotating lobed rings
    for (let ring = 0; ring < 3; ring++) {
      const base = NEST_RADIUS - 3 - ring * 6.5;
      const lobes = 6 + ring * 3;
      const amp = 1.6 + ring * 0.5;
      const rot = t * (ring % 2 === 0 ? 0.35 : -0.5) + ring;
      ctx.strokeStyle = ring === 0 ? 'rgba(0,225,255,0.75)' : ring === 1 ? 'rgba(45,212,191,0.6)' : 'rgba(167,243,208,0.5)';
      ctx.lineWidth = ring === 0 ? 1.4 : 1;
      ctx.beginPath();
      for (let k = 0; k <= 48; k++) {
        const a = (k / 48) * TAU;
        const r = base + Math.sin(a * lobes + rot * lobes) * amp;
        const px = nx + Math.cos(a + rot * 0.3) * r;
        const py = ny + Math.sin(a + rot * 0.3) * r;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // stored food arc: progress toward the next hatch (3 food = 1 ant)
    const hatch = Math.min(1, sim.broodCredit[c] / FOOD_PER_BIRTH);
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = 'rgba(255,168,0,0.22)';
    ctx.beginPath();
    ctx.arc(nx, ny, NEST_RADIUS + 7, 0, TAU);
    ctx.stroke();
    if (hatch > 0) {
      ctx.strokeStyle = 'rgba(255,190,60,0.95)';
      ctx.beginPath();
      ctx.arc(nx, ny, NEST_RADIUS + 7, -Math.PI / 2, -Math.PI / 2 + TAU * hatch);
      ctx.stroke();
    }
    // larval pods: births toward the next colony generation (25 births)
    const pods = 10;
    const lit = ((sim.births % BIRTHS_PER_GENERATION) / BIRTHS_PER_GENERATION) * pods;
    for (let k = 0; k < pods; k++) {
      const a = (k / pods) * TAU - Math.PI / 2 - t * 0.2;
      const px = nx + Math.cos(a) * 9;
      const py = ny + Math.sin(a) * 9;
      const on = k < lit;
      ctx.fillStyle = on ? '#a7f3d0' : 'rgba(167,243,208,0.18)';
      ctx.beginPath();
      ctx.ellipse(px, py, on ? 2.1 + Math.sin(t * 3 + k) * 0.3 : 1.6, on ? 1.5 : 1.1, a, 0, TAU);
      ctx.fill();
    }
    ctx.fillStyle = '#eafff5';
    ctx.beginPath();
    ctx.arc(nx, ny, 2.2 + Math.sin(t * 2) * 0.5, 0, TAU);
    ctx.fill();
    // GEN aura
    const pulse = (t * 0.5) % 1;
    ctx.strokeStyle = `rgba(0,225,255,${(0.35 * (1 - pulse)).toFixed(3)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(nx, ny, NEST_RADIUS + 10 + pulse * 16, 0, TAU);
    ctx.stroke();
    if (this.genCache !== sim.colonyGeneration) {
      this.genCache = sim.colonyGeneration;
      this.genLabel = `GEN ${sim.colonyGeneration}`;
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `700 10px ${this.fontFamily}`;
    ctx.fillStyle = 'rgba(167,243,208,0.9)';
    ctx.fillText(this.genLabel, nx, ny + NEST_RADIUS + 20);
    if (ab) {
      ctx.font = `700 9px ${this.fontFamily}`;
      ctx.fillStyle = c === 0 ? 'rgba(255,210,74,0.95)' : 'rgba(251,113,133,0.95)';
      ctx.fillText(c === 0 ? 'TRAINED BRAIN' : this.rivalLabel(), nx, ny - NEST_RADIUS - 18);
    }
  }

  render(now: number, brush: BrushVisual): void {
    const ctx = this.ctx;
    const sim = this.sim;
    const grid = sim.grid;
    const k = this.scale * this.dpr;
    const ox = this.offX;
    const oy = this.offY;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const W = sim.width;
    const H = sim.height;
    const ab = sim.nestCount === 2;
    const dtMs = this.lastNow > 0 ? Math.min(100, now - this.lastNow) : 16;
    this.lastNow = now;
    this.frame++;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#010204';
    ctx.fillRect(0, 0, cw, ch);
    ctx.save();
    ctx.setTransform(k, 0, 0, k, ox, oy);
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;

    /* 1: static terrain cache */
    this.flushTerrain();
    ctx.drawImage(this.terrain, 0, 0);

    /* 2: full-spectrum pheromone field ('screen') */
    grid.renderRGBA(this.pherImage.data, this.gamma);
    this.pctx.putImageData(this.pherImage, 0, 0);
    const pw = grid.cols * PHEROMONE_CELL;
    const phh = grid.rows * PHEROMONE_CELL;
    ctx.globalCompositeOperation = 'screen';
    ctx.drawImage(this.pher, 0, 0, pw, phh);

    /* 3: 1/4-res prismatic bloom ('lighter'), refreshed every 2nd frame, shed by the adaptive scaler */
    if (!this.adaptive) {
      if ((this.frame & 1) === 0) {
        this.bctx.globalCompositeOperation = 'copy';
        this.bctx.imageSmoothingEnabled = true;
        this.bctx.drawImage(this.pher, 0, 0, this.bloom.width, this.bloom.height);
      }
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 0.55;
      ctx.drawImage(this.bloom, 0, 0, pw, phh);
      ctx.globalAlpha = 1;
    }
    ctx.globalCompositeOperation = 'source-over';

    /* 4: corpse husks + emerald crystals */
    const a = sim.ants;
    const n = sim.count;
    for (let i = 0; i < n; i++) {
      if (a.active[i] === 0 || a.state[i] !== STATE_DEAD) continue;
      const fade = a.link[i] === -2 ? 0.55 : 0.9;
      const ang = a.angle[i];
      const c = Math.cos(ang) * k;
      const s = Math.sin(ang) * k;
      ctx.setTransform(c, s, -s, c, ox + a.x[i] * k, oy + a.y[i] * k);
      ctx.globalAlpha = fade;
      ctx.drawImage(this.husk, 0, 0, HUSK_CELL, HUSK_CELL, -6, -6, 12, 12);
    }
    ctx.globalAlpha = 1;
    ctx.setTransform(k, 0, 0, k, ox, oy);
    const food = sim.food;
    for (let f = 0; f < sim.foodCount; f++) {
      if (food.active[f] === 0) continue;
      const fx = food.x[f];
      const fy = food.y[f];
      if (food.amount[f] >= 1) {
        // crystals shrink smoothly as they are eaten
        const size = food.radius[f] * 2.5 + Math.sin(now * 0.002 + f) * 0.8;
        ctx.drawImage(this.crystalSheet, food.variant[f] * CRYSTAL_CELL, 0, CRYSTAL_CELL, CRYSTAL_CELL, fx - size / 2, fy - size / 2, size, size);
      } else {
        ctx.strokeStyle = 'rgba(80,255,100,0.35)';
        ctx.lineWidth = 1;
        ctx.setLineDash(DASH);
        ctx.beginPath();
        ctx.arc(fx, fy, 9, 0, TAU);
        ctx.stroke();
        ctx.setLineDash(NO_DASH);
      }
    }
    if (ab) {
      for (let c = 0; c < 2; c++) {
        ctx.strokeStyle = 'rgba(122,108,94,0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash(DASH);
        ctx.beginPath();
        ctx.arc(sim.graveX[c], sim.graveY[c], 12, 0, TAU);
        ctx.stroke();
        ctx.setLineDash(NO_DASH);
      }
    } else {
      ctx.strokeStyle = 'rgba(122,108,94,0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash(DASH);
      ctx.beginPath();
      ctx.arc(sim.graveX[0], sim.graveY[0], 12, 0, TAU);
      ctx.stroke();
      ctx.setLineDash(NO_DASH);
    }

    /* 5: brood reactor nest crater */
    for (let c = 0; c < sim.nestCount; c++) this.drawNest(c, now, ab);

    /* 6: articulated ants (batched rotated drawImage) + predators */
    const sheet = ab ? this.cohortSheet : this.casteSheet;
    for (let i = 0; i < n; i++) {
      if (a.active[i] === 0 || a.state[i] === STATE_DEAD) continue;
      const ang = a.angle[i];
      const c = Math.cos(ang) * k;
      const s = Math.sin(ang) * k;
      ctx.setTransform(c, s, -s, c, ox + a.x[i] * k, oy + a.y[i] * k);
      const row = (ab ? a.cohort[i] : a.caste[i]) * 2 + a.carryingFood[i];
      ctx.drawImage(sheet, ((a.legPhase[i] | 0) & 7) * ANT_CELL, row * ANT_CELL, ANT_CELL, ANT_CELL, -7, -7, 14, 14);
    }
    const p = sim.pred;
    for (let q = 0; q < MAX_PREDATORS; q++) {
      if (p.active[q] === 0) continue;
      const ang = p.angle[q];
      const c = Math.cos(ang) * k;
      const s = Math.sin(ang) * k;
      ctx.setTransform(c, s, -s, c, ox + p.x[q] * k, oy + p.y[q] * k);
      ctx.drawImage(this.predatorSheet, ((p.phase[q] | 0) & 7) * PREDATOR_CELL, 0, PREDATOR_CELL, PREDATOR_CELL, -18, -18, 36, 36);
      ctx.setTransform(k, 0, 0, k, ox, oy);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255,12,75,0.2)';
      ctx.beginPath();
      ctx.arc(p.x[q], p.y[q], 22, 0, TAU);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(255,12,75,0.95)';
      ctx.beginPath();
      ctx.arc(p.x[q], p.y[q], 22, -Math.PI / 2, -Math.PI / 2 + TAU * Math.max(0, p.hp[q] / 100));
      ctx.stroke();
    }
    ctx.setTransform(k, 0, 0, k, ox, oy);

    /* 7: synaptic + combat arcs (3-frame vector filaments) */
    const arcs = sim.arcs;
    ctx.lineCap = 'round';
    for (let q = 0; q < MAX_ARCS; q++) {
      const ttl = arcs.ttl[q];
      if (ttl === 0) continue;
      arcs.ttl[q] = ttl - 1;
      const x1 = arcs.x1[q];
      const y1 = arcs.y1[q];
      const x2 = arcs.x2[q];
      const y2 = arcs.y2[q];
      const kind = arcs.kind[q];
      const jit = Math.sin(q * 12.9898 + this.frame * 1.7) * 5;
      const mx = (x1 + x2) / 2 - (y2 - y1) * 0.25 + jit;
      const my = (y1 + y2) / 2 + (x2 - x1) * 0.25 - jit;
      ctx.globalAlpha = ttl / 3;
      ctx.strokeStyle = kind === ARC_COMBAT ? '#ff0c4b' : kind === ARC_TANDEM ? '#af32ff' : '#ffd24a';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(mx, my, x2, y2);
      ctx.stroke();
      ctx.strokeStyle = kind === ARC_COMBAT ? '#ffd5df' : kind === ARC_TANDEM ? '#00e1ff' : '#00e676';
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(mx, my, x2, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    /* 8: atmospheric spore motes (device space, depth-parallax drift) */
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const dts = dtMs * 0.001;
    for (let i = 0; i < MAX_MOTES; i++) {
      const z = this.moteZ[i];
      let mx = this.moteX[i] + dts * 0.006 * z;
      let my = this.moteY[i] - dts * 0.004 * z + Math.sin(now * 0.0004 + this.motePhase[i]) * 0.00012;
      if (mx > 1.02) mx = -0.02;
      if (my < -0.02) my = 1.02;
      this.moteX[i] = mx;
      this.moteY[i] = my;
      const tw = 0.5 + 0.5 * Math.sin(now * 0.0013 + this.motePhase[i]);
      ctx.globalAlpha = (0.06 + 0.2 * z) * (0.4 + 0.6 * tw);
      ctx.fillStyle = i % 3 === 0 ? '#a7f3d0' : i % 3 === 1 ? '#38bdf8' : '#e2e8f0';
      const sz = (0.8 + z * 1.4) * this.dpr;
      ctx.fillRect(mx * cw, my * ch, sz, sz);
    }
    ctx.globalAlpha = 1;

    /* 9: active brush ribbon */
    if (brush.visible) {
      ctx.setTransform(k, 0, 0, k, ox, oy);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const r = brush.radius;
      const tc = brush.trailCount;
      if (tc > 1) {
        const tr = brush.trail;
        for (let pass = 0; pass < 3; pass++) {
          ctx.strokeStyle = pass === 2 ? '#ffffff' : brush.color; // spectral halo, then white-hot core
          ctx.globalAlpha = pass === 0 ? 0.16 : pass === 1 ? 0.42 : 0.92;
          ctx.lineWidth = pass === 0 ? Math.max(5, r * 0.8) : pass === 1 ? Math.max(2.5, r * 0.26) : 1.5;
          ctx.beginPath();
          ctx.moveTo(tr[0], tr[1]);
          for (let q = 1; q < tc; q++) ctx.lineTo(tr[q * 2], tr[q * 2 + 1]);
          ctx.stroke();
        }
      }
      const age = now - brush.startTime;
      if (age >= 0 && age < 520) {
        const u = age / 520;
        ctx.globalAlpha = 1 - u;
        ctx.strokeStyle = brush.color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(brush.startX, brush.startY, r * (0.4 + u * 1.7), 0, TAU);
        ctx.stroke();
      }
      ctx.globalAlpha = brush.down ? 0.95 : 0.5;
      if (brush.erase) {
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1.4;
        ctx.setLineDash(DASH);
        ctx.lineDashOffset = -now * 0.03;
        ctx.beginPath();
        ctx.arc(brush.x, brush.y, r, 0, TAU);
        ctx.stroke();
        ctx.setLineDash(NO_DASH);
        ctx.lineDashOffset = 0;
        ctx.beginPath();
        ctx.moveTo(brush.x - 4, brush.y);
        ctx.lineTo(brush.x + 4, brush.y);
        ctx.moveTo(brush.x, brush.y - 4);
        ctx.lineTo(brush.x, brush.y + 4);
        ctx.stroke();
      } else {
        ctx.strokeStyle = brush.color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(brush.x, brush.y, r, 0, TAU);
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(brush.x, brush.y, 1.6, 0, TAU);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    /* 10: cached vignette + grain */
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.overlay) ctx.drawImage(this.overlay, 0, 0, cw, ch);

    this.noteFrame(dtMs);
  }
}
