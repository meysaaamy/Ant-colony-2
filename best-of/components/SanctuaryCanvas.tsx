/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2.zip/components/SanctuaryCanvas.tsx)
 * Why best: رابط الفابلهی séd interacting با sim: ۶ ابزار، صفحه ۳냅، HUD،_top pills
 * Borrow from other zips: mobile: BioSlider (دابل‌تاپ reset)؛ astramax: D3 dispersion graph، تجربیات /api/health
 * Notes: براي ادجام به best-of/components/SanctuaryCanvas.tsx
 * Status: verbatim copy — adapt import paths when merging into app/
 */

'use client';

import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { ColonySim, FIXED_DT } from '../lib/ant-soa-engine';
import { sampleCatmullRom } from '../lib/pheromone-grid';
import { applyPreset } from '../lib/presets';
import { Renderer, TRAIL_POINTS, createBrushVisual } from '../lib/renderer';
import {
  CHANNEL_CSS, EVENT_FIRST_FOOD, EVENT_FOUNDER, EVENT_GENERATION, EVENT_PREDATOR_SLAIN,
} from '../lib/types';
import type { ArenaMode, PresetId, SubMode, Telemetry, ToastMessage, ToolId } from '../lib/types';

export interface SanctuaryHandle {
  getSim: () => ColonySim | null;
  resetWorld: (preset: PresetId, arena: ArenaMode) => void;
  clearTrails: () => void;
  spawnPredator: () => boolean;
  setPredators: (n: number) => void;
  setBrain: (weights: Float32Array) => void;
}

export interface SanctuaryProps {
  tool: ToolId;
  subMode: SubMode;
  radius: number;
  strength: number;
  paused: boolean;
  speed: number;
  /** Headless work in progress: keep rendering but throttle live physics. */
  busy: boolean;
  initialPreset: PresetId;
  initialArena: ArenaMode;
  onReady: (sim: ColonySim) => void;
  onTelemetry: (t: Telemetry) => void;
  onPointerActive: (active: boolean) => void;
  onEvent: (text: string, tone: ToastMessage['tone']) => void;
}

const STEP_MS = 1000 / 60;
const INITIAL_POP = 800;

/** Fixed world, at least 720 x 1280 (or 1280 x 720), matched to the first viewport's aspect ratio. */
function worldSize(vw: number, vh: number): [number, number] {
  const portrait = vh >= vw;
  const shortV = Math.max(1, Math.min(vw, vh));
  const longV = Math.max(1, Math.max(vw, vh));
  const scale = Math.max(720 / shortV, 1280 / longV);
  const s = Math.min(1040, Math.round(shortV * scale));
  const l = Math.min(1680, Math.round(longV * scale));
  return portrait ? [s, l] : [l, s];
}

function brushColor(tool: ToolId, sub: SubMode): string {
  if (tool === 'crystal') return '#00e676';
  if (tool === 'wall') return '#38bdf8';
  if (tool === 'eraser') return '#e2e8f0';
  return sub === 5 ? CHANNEL_CSS[2] : CHANNEL_CSS[sub];
}

const SanctuaryCanvas = forwardRef<SanctuaryHandle, SanctuaryProps>(function SanctuaryCanvas(props, ref) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simRef = useRef<ColonySim | null>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const propsRef = useRef<SanctuaryProps>(props);
  const seedRef = useRef(20260);

  useEffect(() => {
    propsRef.current = props;
  });

  useImperativeHandle(
    ref,
    () => ({
      getSim: () => simRef.current,
      resetWorld: (preset: PresetId, arena: ArenaMode) => {
        const sim = simRef.current;
        if (!sim) return;
        sim.setArena(arena);
        seedRef.current += 1;
        applyPreset(sim, preset, seedRef.current, INITIAL_POP);
        rendererRef.current?.invalidateTerrain();
      },
      clearTrails: () => {
        simRef.current?.grid.clearPheromones();
      },
      spawnPredator: () => {
        const sim = simRef.current;
        if (!sim) return false;
        const before = sim.predatorCount;
        sim.setPredatorCount(before + 1);
        return sim.predatorCount > before;
      },
      setPredators: (n: number) => {
        simRef.current?.setPredatorCount(n);
      },
      setBrain: (weights: Float32Array) => {
        simRef.current?.setBrain(weights);
      },
    }),
    [],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const [ww, wh] = worldSize(window.innerWidth, window.innerHeight);
    const sim = new ColonySim(ww, wh, seedRef.current);
    sim.setArena(propsRef.current.initialArena);
    applyPreset(sim, propsRef.current.initialPreset, seedRef.current, INITIAL_POP);
    const renderer = new Renderer(canvas, sim, 4242);
    renderer.setFont(getComputedStyle(document.body).fontFamily);
    renderer.resize(canvas.clientWidth, canvas.clientHeight, window.devicePixelRatio || 1);
    simRef.current = sim;
    rendererRef.current = renderer;
    propsRef.current.onReady(sim);

    const brush = createBrushVisual();
    const world = new Float32Array(2);
    const stamps = new Float32Array(256);
    // last three stroke samples for the Catmull-Rom segment
    let p0x = 0, p0y = 0, p1x = 0, p1y = 0, p2x = 0, p2y = 0;
    let lastCrystalX = -1e9;
    let lastCrystalY = -1e9;
    let activePointer: number | null = null;
    let hidden = document.visibilityState === 'hidden';
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    let resizeTimer: ReturnType<typeof setTimeout> | null = null;

    const pushTrail = (x: number, y: number): void => {
      const tr = brush.trail;
      if (brush.trailCount >= TRAIL_POINTS) {
        tr.copyWithin(0, 2);
        brush.trailCount = TRAIL_POINTS - 1;
      }
      tr[brush.trailCount * 2] = x;
      tr[brush.trailCount * 2 + 1] = y;
      brush.trailCount++;
    };

    const applyTool = (x: number, y: number, first: boolean): void => {
      const p = propsRef.current;
      const r = p.radius;
      if (p.tool === 'food-trail' || p.tool === 'home-trail' || p.tool === 'alarm') {
        if (p.subMode === 5) {
          if (first && !sim.spawnPredator(x, y)) p.onEvent('Predator cap reached (6) or blocked by basalt', 'warn');
        } else {
          sim.grid.stampDisc(p.subMode, x, y, r, 9 * p.strength);
        }
      } else if (p.tool === 'crystal') {
        const dx = x - lastCrystalX;
        const dy = y - lastCrystalY;
        if (first || dx * dx + dy * dy > 44 * 44) {
          if (!sim.grid.isWallAt(x, y)) sim.addFood(x, y, Math.round(40 + r * 2.5));
          lastCrystalX = x;
          lastCrystalY = y;
        }
      } else if (p.tool === 'wall') {
        const wr = Math.max(7, r * 0.55);
        if (sim.paintWall(x, y, wr, true) > 0) renderer.markTerrainDirty(x, y, wr);
      } else if (sim.eraseAt(x, y, r) > 0) {
        renderer.markTerrainDirty(x, y, r);
      }
    };

    const toWorld = (e: PointerEvent): void => {
      const rect = canvas.getBoundingClientRect();
      renderer.toWorld(e.clientX - rect.left, e.clientY - rect.top, world);
    };

    const onDown = (e: PointerEvent): void => {
      if (activePointer !== null) return; // single pointerId
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      activePointer = e.pointerId;
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        /* capture is best-effort */
      }
      e.preventDefault();
      toWorld(e);
      const x = world[0];
      const y = world[1];
      p0x = p1x = p2x = x;
      p0y = p1y = p2y = y;
      brush.visible = true;
      brush.down = true;
      brush.x = x;
      brush.y = y;
      brush.startX = x;
      brush.startY = y;
      brush.startTime = performance.now();
      brush.trailCount = 0;
      pushTrail(x, y);
      applyTool(x, y, true);
      propsRef.current.onPointerActive(true);
    };

    const onMove = (e: PointerEvent): void => {
      if (activePointer === null) {
        if (e.pointerType === 'mouse') {
          toWorld(e);
          brush.visible = true;
          brush.x = world[0];
          brush.y = world[1];
        }
        return;
      }
      if (e.pointerId !== activePointer) return;
      toWorld(e);
      const x = world[0];
      const y = world[1];
      const dx = x - p2x;
      const dy = y - p2y;
      if (dx * dx + dy * dy < 1) return;
      // Catmull-Rom through the last samples, stamped every 0.25 * radius
      const n = sampleCatmullRom(p0x, p0y, p1x, p1y, x, y, x + (x - p1x) * 0.5, y + (y - p1y) * 0.5,
        propsRef.current.radius * 0.25, stamps);
      // the sampled segment runs p1 -> current; re-anchor so consecutive segments join smoothly
      for (let k = 0; k < n; k++) applyTool(stamps[k * 2], stamps[k * 2 + 1], false);
      p0x = p1x;
      p0y = p1y;
      p1x = x;
      p1y = y;
      p2x = x;
      p2y = y;
      brush.x = x;
      brush.y = y;
      pushTrail(x, y);
    };

    const onUp = (e: PointerEvent): void => {
      if (e.pointerId !== activePointer) return;
      activePointer = null;
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
      brush.down = false;
      if (e.pointerType !== 'mouse') brush.visible = false;
      propsRef.current.onPointerActive(false);
    };

    const onLeave = (): void => {
      if (activePointer === null) brush.visible = false;
    };

    const onVisibility = (): void => {
      hidden = document.visibilityState === 'hidden';
      last = performance.now();
      acc = 0;
    };

    const onResize = (): void => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        renderer.resize(canvas.clientWidth, canvas.clientHeight, window.devicePixelRatio || 1);
      }, 150);
    };

    const frame = (now: number): void => {
      raf = requestAnimationFrame(frame);
      const p = propsRef.current;
      let dt = now - last;
      last = now;
      if (dt > 100) dt = 100;
      if (!p.paused && !hidden) {
        acc += dt * p.speed;
        const maxSteps = p.busy ? 1 : 8;
        let steps = 0;
        while (acc >= STEP_MS && steps < maxSteps) {
          sim.step(FIXED_DT);
          acc -= STEP_MS;
          steps++;
        }
        if (acc > STEP_MS * 4) acc = 0;
      }
      brush.radius = p.tool === 'wall' ? Math.max(7, p.radius * 0.55) : p.radius;
      brush.erase = p.tool === 'eraser';
      brush.color = brushColor(p.tool, p.subMode);
      if (!brush.down && brush.trailCount > 0) {
        // ribbon retracts after release
        const drop = Math.min(brush.trailCount, 3);
        brush.trail.copyWithin(0, drop * 2);
        brush.trailCount -= drop;
      }
      if (!hidden) renderer.render(now, brush);
    };

    const telemetry = setInterval(() => {
      const p = propsRef.current;
      const ev = sim.events;
      if (ev !== 0) {
        sim.events = 0;
        if (ev & EVENT_FOUNDER) p.onEvent('Founder Bottleneck: emergency brood wave, 2.5x mutation boost armed', 'warn');
        else if (ev & EVENT_GENERATION) p.onEvent(`Colony generation ${sim.colonyGeneration} has hatched`, 'good');
        if (ev & EVENT_PREDATOR_SLAIN) p.onEvent('Wardens neutralised an Abyssal Stalker', 'good');
        if (ev & EVENT_FIRST_FOOD) p.onEvent('First emerald delivered to the brood reactor', 'good');
      }
      p.onTelemetry({
        fps: Math.round(renderer.fps),
        alive: sim.alive,
        deaths: sim.totalDeaths,
        food: sim.storedFood,
        generation: sim.colonyGeneration,
        births: sim.births,
        predators: sim.predatorCount,
        scoreA: sim.delivered[0],
        scoreB: sim.delivered[1],
        arena: sim.arena,
        adaptive: renderer.adaptive,
      });
    }, 100);

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('pointerleave', onLeave);
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('resize', onResize);
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      clearInterval(telemetry);
      if (resizeTimer) clearTimeout(resizeTimer);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      canvas.removeEventListener('pointerleave', onLeave);
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('resize', onResize);
      simRef.current = null;
      rendererRef.current = null;
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="sanctuary-canvas"
      role="img"
      aria-label="Live ant colony simulation. Draw on the canvas with the selected tool."
    />
  );
});

export default SanctuaryCanvas;
