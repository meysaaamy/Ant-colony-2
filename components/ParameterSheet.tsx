'use client';

import { useEffect, useRef, useState } from 'react';
import type { ComponentType, CSSProperties, ReactNode, SVGProps } from 'react';
import { ColonySim, NEST_RADIUS } from '../lib/ant-soa-engine';
import { DEFAULT_GAMMA, PHEROMONE_CELL, mixPheromoneRGB } from '../lib/pheromone-grid';
import { PRESETS, applyPreset } from '../lib/presets';
import { CHANNEL_CSS, CHANNEL_KEYS, CHANNEL_NAMES } from '../lib/types';
import type { PresetId } from '../lib/types';
import { ChartIcon, FlaskIcon, GlobeIcon, PredatorIcon, RadiusIcon, ResetIcon, SkullIcon } from './ui/icons';

export type SheetSnap = 'peek' | 'half' | 'full';
export type SheetTab = 'proof' | 'colony' | 'alchemy' | 'brush' | 'world';

export interface Settings {
  maxPop: number;
  atpBurn: number;
  lifespan: number;
  predators: number;
  evaporation: number;
  diffusion: number;
  crossReaction: number;
  radius: number;
  strength: number;
}

type IconType = ComponentType<SVGProps<SVGSVGElement> & { size?: number }>;

const TABS: ReadonlyArray<{ id: SheetTab; label: string; Icon: IconType }> = [
  { id: 'proof', label: 'Proof of Learning', Icon: ChartIcon },
  { id: 'colony', label: 'Colony & Threats', Icon: SkullIcon },
  { id: 'alchemy', label: 'Pheromone Alchemy', Icon: FlaskIcon },
  { id: 'brush', label: 'Brush', Icon: RadiusIcon },
  { id: 'world', label: 'World', Icon: GlobeIcon },
];

const SNAPS: readonly SheetSnap[] = ['peek', 'half', 'full'];

interface SheetProps {
  snap: SheetSnap;
  onSnap: (s: SheetSnap) => void;
  tab: SheetTab;
  onTab: (t: SheetTab) => void;
  presetId: PresetId;
  onPreset: (id: PresetId) => void;
  settings: Settings;
  onSetting: (key: keyof Settings, value: number) => void;
  livePredators: number;
  abRandomActive: boolean;
  busy: boolean;
  onTrain5: () => void;
  onAbRandom: () => void;
  onPredatorThreat: () => void;
  onClearTrails: () => void;
  onReset: () => void;
  children: ReactNode;
}

interface FieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}

function Field({ label, value, min, max, step, display, onChange }: FieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <output className="tnum">{display}</output>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

/** 5-channel RGB mixing preview rendered through the exact mixer the simulation uses. */
function MixPreview() {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.width;
    const h = c.height;
    const img = ctx.createImageData(w, h);
    const g = (x: number, y: number, cx: number, cy: number, r: number): number => {
      const dx = (x - cx) / r;
      const dy = (y - cy) / r;
      return Math.exp(-(dx * dx + dy * dy));
    };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const u = x / w;
        const v = y / h;
        const band = (cy: number, s: number): number => Math.exp(-((v - cy) / s) * ((v - cy) / s));
        const food = (u > 0.04 && u < 0.66 ? 62 : 0) * band(0.34, 0.11);
        const home = (u > 0.3 && u < 0.9 ? 54 : 0) * band(0.38, 0.13);
        const scout = 70 * g(u, v, 0.13, 0.5, 0.11);
        const alarm = 78 * g(u, v, 0.82, 0.62, 0.13);
        const noEntry = 66 * g(u, v, 0.66, 0.82, 0.1);
        const o = (y * w + x) * 4;
        mixPheromoneRGB(food, home, alarm, scout, noEntry, DEFAULT_GAMMA, img.data, o);
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, []);
  return (
    <canvas
      ref={ref}
      width={170}
      height={72}
      style={{ width: '100%', borderRadius: 12, background: '#020307' }}
      role="img"
      aria-label="Preview of the five pheromone channels mixing: pearl highway, rose-gold, magenta plume and sulfur-gold hazard"
    />
  );
}

/** Mini canvas previews for the three presets, built with the real preset generators. */
function PresetPreviews({ presetId, onPreset }: { presetId: PresetId; onPreset: (id: PresetId) => void }) {
  const refs = useRef<Array<HTMLCanvasElement | null>>([]);
  useEffect(() => {
    const sim = new ColonySim(480, 360, 5);
    sim.setArena('trained');
    PRESETS.forEach((p, idx) => {
      const c = refs.current[idx];
      if (!c) return;
      const ctx = c.getContext('2d');
      if (!ctx) return;
      applyPreset(sim, p.id, 11, 0);
      const k = c.width / 480;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = '#020307';
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.setTransform(k, 0, 0, k, 0, 0);
      const grid = sim.grid;
      for (let pass = 0; pass < 2; pass++) {
        ctx.fillStyle = pass === 0 ? '#2dd4bf' : '#0d1322';
        for (let gy = 0; gy < grid.rows; gy++) {
          for (let gx = 0; gx < grid.cols; gx++) {
            if (grid.wall[gy * grid.cols + gx] === 0) continue;
            ctx.beginPath();
            ctx.arc((gx + 0.5) * PHEROMONE_CELL, (gy + 0.5) * PHEROMONE_CELL, PHEROMONE_CELL * (pass === 0 ? 1.15 : 0.85), 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      for (let f = 0; f < sim.foodCount; f++) {
        if (sim.food.active[f] === 0) continue;
        const fx = sim.food.x[f];
        const fy = sim.food.y[f];
        ctx.fillStyle = '#00e676';
        ctx.beginPath();
        ctx.moveTo(fx, fy - 15);
        ctx.lineTo(fx + 10, fy);
        ctx.lineTo(fx, fy + 15);
        ctx.lineTo(fx - 10, fy);
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = '#00e1ff';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.arc(sim.nestX[0], sim.nestY[0], NEST_RADIUS * 0.8, 0, Math.PI * 2);
      ctx.stroke();
    });
  }, []);
  return (
    <div className="flex flex-col gap-2" role="radiogroup" aria-label="World preset">
      {PRESETS.map((p, idx) => (
        <button
          key={p.id}
          type="button"
          role="radio"
          aria-checked={presetId === p.id}
          className="preset-card"
          data-active={presetId === p.id}
          onClick={() => onPreset(p.id)}
        >
          <canvas
            ref={(el) => {
              refs.current[idx] = el;
            }}
            width={192}
            height={144}
            aria-hidden="true"
          />
          <span className="flex flex-col gap-1">
            <span className="font-display text-[15px] leading-5 font-semibold">{p.title}</span>
            <span className="section-note">{p.blurb}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

/** Tier 1 peek card that expands into the 3-snap sheet (390px side panel on desktop). */
export default function ParameterSheet(props: SheetProps) {
  const {
    snap, onSnap, tab, onTab, presetId, onPreset, settings, onSetting, livePredators, abRandomActive, busy,
    onTrain5, onAbRandom, onPredatorThreat, onClearTrails, onReset, children,
  } = props;
  const [armed, setArmed] = useState(false);
  const armTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const drag = useRef<{ id: number; y: number } | null>(null);
  const title = PRESETS.find((p) => p.id === presetId)?.title ?? PRESETS[0].title;

  useEffect(
    () => () => {
      if (armTimer.current) clearTimeout(armTimer.current);
    },
    [],
  );

  const stepSnap = (dir: number): void => {
    const i = SNAPS.indexOf(snap);
    const n = Math.max(0, Math.min(SNAPS.length - 1, i + dir));
    onSnap(SNAPS[n]);
  };

  const handleReset = (): void => {
    if (armed) {
      if (armTimer.current) clearTimeout(armTimer.current);
      setArmed(false);
      onReset();
      return;
    }
    setArmed(true);
    armTimer.current = setTimeout(() => setArmed(false), 2600);
  };

  return (
    <section className="sheet glass" data-snap={snap} aria-label="Colony cockpit">
      <button
        type="button"
        className="sheet-handle"
        aria-label={snap === 'full' ? 'Collapse sheet' : 'Expand sheet'}
        onPointerDown={(e) => {
          drag.current = { id: e.pointerId, y: e.clientY };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d || d.id !== e.pointerId) return;
          const dy = e.clientY - d.y;
          if (dy < -28) stepSnap(1);
          else if (dy > 28) stepSnap(-1);
          else onSnap(snap === 'peek' ? 'half' : snap === 'half' ? 'full' : 'peek');
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            e.stopPropagation();
            onSnap(snap === 'peek' ? 'half' : snap === 'half' ? 'full' : 'peek');
          }
        }}
      >
        <span />
      </button>
      <div className="peek">
        <div className="flex items-center justify-between gap-3">
          <h1 className="peek-title font-display">{title}</h1>
          <span className="badge" style={{ flex: 'none' }}>{livePredators} / 6 stalkers</span>
        </div>
        <div className="pill-row">
          <button type="button" className="pill" style={{ '--pill-color': '#00e676' } as CSSProperties} disabled={busy} onClick={onTrain5}>
            <span className="dot" />
            Train +5 Gen
          </button>
          <button
            type="button"
            className="pill"
            style={{ '--pill-color': '#ffd24a' } as CSSProperties}
            data-active={abRandomActive}
            aria-pressed={abRandomActive}
            onClick={onAbRandom}
          >
            <span className="dot" />
            A/B vs Random
          </button>
          <button type="button" className="pill" style={{ '--pill-color': CHANNEL_CSS[2] } as CSSProperties} onClick={onPredatorThreat}>
            <PredatorIcon size={16} style={{ color: CHANNEL_CSS[2] }} />
            Predator Threat
          </button>
          <button type="button" className="pill" onClick={onClearTrails}>Clear trails</button>
          <button type="button" className="pill" data-tone="danger" data-armed={armed} onClick={handleReset} aria-live="polite">
            <ResetIcon size={16} />
            {armed ? 'Tap again to reset' : 'Reset'}
          </button>
        </div>
      </div>
      {snap !== 'peek' && (
        <>
          <div className="tabs" role="tablist" aria-label="Cockpit sections">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={tab === t.id}
                className="tab"
                data-active={tab === t.id}
                onClick={() => onTab(t.id)}
              >
                <t.Icon size={16} />
                {t.label}
              </button>
            ))}
          </div>
          <div className="sheet-body" role="tabpanel">
            {tab === 'proof' && children}

            {tab === 'colony' && (
              <>
                <h3 className="section-title font-display">Colony and threats</h3>
                <div className="card">
                  <Field label="Max population" value={settings.maxPop} min={100} max={3000} step={50}
                    display={String(settings.maxPop)} onChange={(v) => onSetting('maxPop', v)} />
                  <Field label="ATP burn" value={settings.atpBurn} min={0.25} max={3} step={0.05}
                    display={`${settings.atpBurn.toFixed(2)}x`} onChange={(v) => onSetting('atpBurn', v)} />
                  <Field label="Lifespan" value={settings.lifespan} min={60} max={600} step={10}
                    display={`${settings.lifespan}s`} onChange={(v) => onSetting('lifespan', v)} />
                </div>
                <div className="card">
                  <Field label="Predator count" value={settings.predators} min={0} max={6} step={1}
                    display={`${settings.predators} (live ${livePredators})`} onChange={(v) => onSetting('predators', v)} />
                  <button type="button" className="btn ghost" onClick={onPredatorThreat}>
                    <PredatorIcon size={18} style={{ color: CHANNEL_CSS[2] }} />
                    Spawn Predator
                  </button>
                  <p className="section-note">
                    Abyssal Stalkers hunt the nearest ant. Ants within 75 px broadcast a crimson alarm burst; Wardens
                    climb it and wear the predator down. Every 3 delivered food hatches 1 ant; 25 births advance the
                    colony generation. Under 35 survivors a founder brood wave is released.
                  </p>
                </div>
              </>
            )}

            {tab === 'alchemy' && (
              <>
                <h3 className="section-title font-display">Pheromone alchemy</h3>
                <MixPreview />
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {CHANNEL_NAMES.map((n, i) => (
                    <span key={n} className="inline-flex items-center gap-1.5 text-[11px] text-[#9aa8bf]">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: CHANNEL_CSS[i] }} />
                      {n} <span className="text-[#64728a]">{CHANNEL_KEYS[i]}</span>
                    </span>
                  ))}
                </div>
                <div className="card">
                  <Field label="Evaporation" value={settings.evaporation} min={0.25} max={3} step={0.05}
                    display={`${settings.evaporation.toFixed(2)}x`} onChange={(v) => onSetting('evaporation', v)} />
                  <Field label="Cross-reaction rate" value={settings.crossReaction} min={0} max={2} step={0.05}
                    display={`${settings.crossReaction.toFixed(2)}x`} onChange={(v) => onSetting('crossReaction', v)} />
                  <Field label="Diffusion" value={settings.diffusion} min={0} max={8} step={0.25}
                    display={settings.diffusion.toFixed(2)} onChange={(v) => onSetting('diffusion', v)} />
                </div>
                <p className="section-note">
                  Food + Home polymerise into pearl highways with turquoise shoulders that decay 35% slower. Food + Scout
                  reads rose-gold, Home + Alarm electric magenta, Alarm + No-Entry sulfur-gold. No-Entry dissolves Food
                  trail at 0.45 per unit per second. Alarm evaporates 3.2x faster than Food.
                </p>
              </>
            )}

            {tab === 'brush' && (
              <>
                <h3 className="section-title font-display">Brush</h3>
                <div className="card">
                  <Field label="Radius" value={settings.radius} min={8} max={64} step={2}
                    display={`${settings.radius}px`} onChange={(v) => onSetting('radius', v)} />
                  <Field label="Strength" value={settings.strength} min={0.3} max={2} step={0.1}
                    display={`${settings.strength.toFixed(1)}x`} onChange={(v) => onSetting('strength', v)} />
                </div>
                <p className="section-note">
                  Strokes are Catmull-Rom splines stamped every quarter radius, so fast drags stay continuous. Pheromone
                  tools take any of the five channels from the pills above the dock; the Alarm tool can also spawn a
                  predator. Walls repaint only the touched region of the terrain cache.
                </p>
              </>
            )}

            {tab === 'world' && (
              <>
                <h3 className="section-title font-display">World</h3>
                <PresetPreviews presetId={presetId} onPreset={onPreset} />
                <p className="section-note">Choosing a preset rebuilds the world and restarts the colony with the current brain.</p>
              </>
            )}
          </div>
        </>
      )}
    </section>
  );
}
