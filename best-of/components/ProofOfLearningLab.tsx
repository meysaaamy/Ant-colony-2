/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2.zip/components/ProofOfLearningLab.tsx)
 * Why best: رابط الفابلهی séd interacting با sim: ۶ ابزار، صفحه ۳냅، HUD،_top pills
 * Borrow from other zips: mobile: BioSlider (دابل‌تاپ reset)؛ astramax: D3 dispersion graph، تجربیات /api/health
 * Notes: براي ادجام به best-of/components/ProofOfLearningLab.tsx
 * Status: verbatim copy — adapt import paths when merging into app/
 */

'use client';

import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import { ABLATIONS } from '../lib/headless-sim';
import { BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS, OFF_W2 } from '../lib/neat';
import { CHANNEL_CSS } from '../lib/types';
import type { AblationId, ArenaMode, BenchmarkProgress, BenchmarkResult, BenchmarkRow, GenerationRecord } from '../lib/types';
import { BrainIcon } from './ui/icons';

export interface TrainingState {
  running: boolean;
  done: number;
  target: number;
  progress: number;
}

interface LabProps {
  history: GenerationRecord[];
  bakedGenerations: number;
  trainedAt: string;
  training: TrainingState;
  onTrain: (generations: number) => void;
  benchmark: BenchmarkResult | null;
  benchProgress: BenchmarkProgress | null;
  onRunBenchmark: () => void;
  arena: ArenaMode;
  onArena: (mode: ArenaMode) => void;
  muted: Record<AblationId, boolean>;
  onToggleMute: (id: AblationId) => void;
  weights: Float32Array;
  weightsVersion: number;
  onExport: () => void;
  onImport: (file: File) => void;
}

const ABLATION_COLOR: Record<string, string> = {
  foodTrail: CHANNEL_CSS[0],
  homeGuide: CHANNEL_CSS[1],
  alarm: CHANNEL_CSS[2],
  scoutPulse: CHANNEL_CSS[3],
  noEntry: CHANNEL_CSS[4],
  pathIntegration: '#cbd5e1',
  trained: '#ffd24a',
  aco: '#00e1ff',
  random: '#fb7185',
};

const ARENAS: ReadonlyArray<{ id: ArenaMode; label: string }> = [
  { id: 'trained', label: '100% Trained Brain' },
  { id: 'ab-random', label: '50/50 Trained vs Random' },
  { id: 'ab-aco', label: '50/50 Trained vs Hand-Crafted ACO' },
  { id: 'aco', label: '100% Hand-Crafted ACO' },
];

function linePath(points: Array<[number, number]>): string {
  let d = '';
  for (let i = 0; i < points.length; i++) d += `${i === 0 ? 'M' : 'L'}${points[i][0].toFixed(1)} ${points[i][1].toFixed(1)}`;
  return d;
}

/** Multi-seed SVG learning curve: Food Delivered per Generation vs ACO baseline and Random floor. */
function LearningCurve({ history, bakedGenerations }: { history: GenerationRecord[]; bakedGenerations: number }) {
  const W = 340;
  const H = 184;
  const L = 34;
  const R = 10;
  const T = 10;
  const B = 26;
  if (history.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="No training history yet">
        <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="12" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)" />
        <text x={W / 2} y={H / 2 - 6} textAnchor="middle" fontSize="12" fill="#9aa8bf">No generations recorded yet</text>
        <text x={W / 2} y={H / 2 + 12} textAnchor="middle" fontSize="11" fill="#64728a">Run Train to evolve the shared brain</text>
      </svg>
    );
  }
  const g0 = history[0].generation;
  const g1 = Math.max(g0 + 1, history[history.length - 1].generation);
  let maxY = 10;
  for (const h of history) maxY = Math.max(maxY, h.bestFood, h.championFood, h.acoFood, h.randomFood);
  maxY *= 1.12;
  const sx = (g: number): number => L + ((g - g0) / (g1 - g0)) * (W - L - R);
  const sy = (v: number): number => H - B - (v / maxY) * (H - T - B);
  const best = linePath(history.map((h) => [sx(h.generation), sy(h.bestFood)]));
  const champ = linePath(history.map((h) => [sx(h.generation), sy(h.championFood)]));
  const aco = linePath(history.map((h) => [sx(h.generation), sy(h.acoFood)]));
  const rnd = linePath(history.map((h) => [sx(h.generation), sy(h.randomFood)]));
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const last = history[history.length - 1];
  const liveStart = bakedGenerations > g0 && bakedGenerations <= g1 ? sx(bakedGenerations) : -1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Learning curve. Champion ${last.championFood.toFixed(0)} food, ACO ${last.acoFood.toFixed(0)}, Random ${last.randomFood.toFixed(0)}.`}>
      <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="12" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(maxY * t)} y2={sy(maxY * t)} stroke="rgba(255,255,255,0.07)" />
          <text x={L - 5} y={sy(maxY * t) + 3} textAnchor="end" fontSize="9" fill="#64728a">{Math.round(maxY * t)}</text>
        </g>
      ))}
      {liveStart > 0 && (
        <g>
          <line x1={liveStart} x2={liveStart} y1={T} y2={H - B} stroke="#af32ff" strokeOpacity="0.6" strokeDasharray="2 3" />
          <text x={liveStart + 4} y={T + 9} fontSize="9" fill="#d8b4fe">in-app</text>
        </g>
      )}
      <path d={rnd} fill="none" stroke="#fb7185" strokeWidth="1.4" strokeDasharray="2 4" />
      <path d={aco} fill="none" stroke="#00e1ff" strokeWidth="1.6" strokeDasharray="6 4" />
      <path d={best} fill="none" stroke="#ffd24a" strokeOpacity="0.38" strokeWidth="1.1" />
      <path d={champ} fill="none" stroke="#ffd24a" strokeWidth="2.2" strokeLinejoin="round" />
      <circle cx={sx(last.generation)} cy={sy(last.championFood)} r="3.2" fill="#ffd24a" />
      <text x={L} y={H - 8} fontSize="9" fill="#64728a">gen {g0}</text>
      <text x={W - R} y={H - 8} textAnchor="end" fontSize="9" fill="#64728a">gen {g1}</text>
      <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" fontSize="9" fill="#9aa8bf">food delivered per episode (mean of 3 seeds)</text>
    </svg>
  );
}

function Legend() {
  const items: Array<{ c: string; t: string; dash?: boolean }> = [
    { c: '#ffd24a', t: 'Shared MLP brain (champion)' },
    { c: '#00e1ff', t: 'Hand-Crafted ACO', dash: true },
    { c: '#fb7185', t: 'Random network', dash: true },
  ];
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[#9aa8bf]">
      {items.map((i) => (
        <span key={i.t} className="inline-flex items-center gap-1.5">
          <svg width="18" height="6" aria-hidden="true">
            <line x1="0" y1="3" x2="18" y2="3" stroke={i.c} strokeWidth="2" strokeDasharray={i.dash ? '4 3' : undefined} />
          </svg>
          {i.t}
        </span>
      ))}
    </div>
  );
}

/** Live SVG bar chart: mean +/- std food per condition, ablations annotated with their delta. */
function BenchmarkChart({ result }: { result: BenchmarkResult }) {
  const W = 340;
  const rowH = 22;
  const L = 104;
  const R = 78;
  const H = result.rows.length * rowH + 12;
  let max = 1;
  for (const r of result.rows) max = Math.max(max, r.mean + r.std);
  const sx = (v: number): number => (Math.max(0, v) / max) * (W - L - R);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Benchmark: food delivered per condition">
      {result.rows.map((r: BenchmarkRow, i: number) => {
        const y = 6 + i * rowH;
        const color = ABLATION_COLOR[r.id] ?? '#cbd5e1';
        const w = sx(r.mean);
        return (
          <g key={r.id}>
            {i === 3 && <line x1="0" x2={W} y1={y - 2} y2={y - 2} stroke="rgba(255,255,255,0.1)" strokeDasharray="2 4" />}
            <text x={L - 8} y={y + 13} textAnchor="end" fontSize="10" fontWeight={r.kind === 'policy' ? 700 : 500} fill={r.kind === 'policy' ? '#e6edf7' : '#9aa8bf'}>
              {r.label}
            </text>
            <rect x={L} y={y + 4} width={W - L - R} height={12} rx="4" fill="rgba(255,255,255,0.045)" />
            <rect x={L} y={y + 4} width={Math.max(1.5, w)} height={12} rx="4" fill={color} fillOpacity={r.kind === 'policy' ? 0.95 : 0.7} />
            <line x1={L + sx(r.mean - r.std)} x2={L + sx(r.mean + r.std)} y1={y + 10} y2={y + 10} stroke="#ffffff" strokeOpacity="0.85" strokeWidth="1.2" />
            <line x1={L + sx(r.mean + r.std)} x2={L + sx(r.mean + r.std)} y1={y + 6.5} y2={y + 13.5} stroke="#ffffff" strokeOpacity="0.85" strokeWidth="1.2" />
            <text x={W - R + 6} y={y + 13} fontSize="10" fill="#e6edf7">
              {r.mean.toFixed(1)}
              <tspan fill="#64728a"> {'\u00B1'}{r.std.toFixed(0)}</tspan>
              {r.kind === 'ablation' && (
                <tspan fill={r.delta < 0 ? '#fb7185' : '#5dffa8'}> {r.delta >= 0 ? '+' : ''}{r.delta.toFixed(0)}</tspan>
              )}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Compact weight heatmap: input->hidden (19 x 16) and hidden->output (16 x 9). Cyan positive, crimson negative. */
function WeightHeatmap({ weights, version }: { weights: Float32Array; version: number }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const cell = 6;
    ctx.fillStyle = '#020307';
    ctx.fillRect(0, 0, c.width, c.height);
    const paint = (w: number, x: number, y: number): void => {
      const m = Math.min(1, Math.abs(w) / 2.5);
      ctx.fillStyle = w >= 0 ? `rgba(0,225,255,${(0.08 + 0.92 * m).toFixed(3)})` : `rgba(255,12,75,${(0.08 + 0.92 * m).toFixed(3)})`;
      ctx.fillRect(x, y, cell - 1, cell - 1);
    };
    for (let h = 0; h < BRAIN_HIDDEN; h++) {
      for (let i = 0; i < BRAIN_INPUTS; i++) paint(weights[h * BRAIN_INPUTS + i], 2 + i * cell, 2 + h * cell);
    }
    const x0 = 2 + BRAIN_INPUTS * cell + 10;
    for (let o = 0; o < BRAIN_OUTPUTS; o++) {
      for (let h = 0; h < BRAIN_HIDDEN; h++) paint(weights[OFF_W2 + o * BRAIN_HIDDEN + h], x0 + o * cell, 2 + h * cell);
    }
  }, [weights, version]);
  return (
    <canvas
      ref={ref}
      width={2 + BRAIN_INPUTS * 6 + 10 + BRAIN_OUTPUTS * 6 + 2}
      height={2 + BRAIN_HIDDEN * 6 + 2}
      style={{ width: '100%', maxWidth: 360, imageRendering: 'pixelated', borderRadius: 12 }}
      role="img"
      aria-label="Weight heatmap of the shared colony brain"
    />
  );
}

export default function ProofOfLearningLab(props: LabProps) {
  const {
    history, bakedGenerations, trainedAt, training, onTrain, benchmark, benchProgress, onRunBenchmark,
    arena, onArena, muted, onToggleMute, weights, weightsVersion, onExport, onImport,
  } = props;
  const fileRef = useRef<HTMLInputElement | null>(null);
  const busy = training.running || benchProgress !== null;
  const last = history.length > 0 ? history[history.length - 1] : null;
  const trainPct = training.running ? ((training.done + training.progress) / Math.max(1, training.target)) * 100 : 0;
  return (
    <>
      <section className="flex flex-col gap-2">
        <h3 className="section-title font-display">Food delivered per generation</h3>
        <p className="section-note">
          One shared 19-16-9 recurrent brain drives every trained ant. The curve is its food delivery in headless
          episodes, averaged over 3 fixed seeds, against two reference policies scored on the same seeds.
        </p>
        <LearningCurve history={history} bakedGenerations={bakedGenerations} />
        <Legend />
        {last && (
          <p className="section-note tnum">
            Generation {last.generation}: champion {last.championFood.toFixed(1)}, ACO {last.acoFood.toFixed(1)}, random {last.randomFood.toFixed(1)}.
            {bakedGenerations > 0 ? ` Shipped brain: ${bakedGenerations} offline generations${trainedAt ? `, baked ${trainedAt}` : ''}.` : ' No offline champion is baked in yet.'}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn" disabled={busy} onClick={() => onTrain(5)}>Train +5 Gen</button>
          <button type="button" className="btn ghost" disabled={busy} onClick={() => onTrain(10)}>+10 Gen</button>
        </div>
        {training.running && (
          <div className="flex flex-col gap-1" role="status">
            <div className="progress"><span style={{ width: `${trainPct.toFixed(1)}%` }} /></div>
            <span className="section-note tnum">
              Evolving generation {training.done + 1} of {training.target}. The live view is throttled while episodes run.
            </span>
          </div>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="section-title font-display">Headless benchmark and ablations</h3>
        <p className="section-note">
          Same ants, same worlds, 3 held-out seeds per condition. Ablation rows mute one input of the trained brain;
          a negative delta means the brain depends on it.
        </p>
        <button type="button" className="btn ghost" disabled={busy} onClick={onRunBenchmark}>
          {benchProgress ? `Running ${benchProgress.done} / ${benchProgress.total}` : 'Run multi-seed benchmark (N = 3)'}
        </button>
        {benchProgress && (
          <div className="flex flex-col gap-1" role="status">
            <div className="progress"><span style={{ width: `${((benchProgress.done / benchProgress.total) * 100).toFixed(1)}%` }} /></div>
            <span className="section-note">{benchProgress.label}</span>
          </div>
        )}
        {benchmark ? (
          <>
            <BenchmarkChart result={benchmark} />
            <span className="section-note tnum">
              Bars: mean food delivered, whiskers: standard deviation across seeds [{benchmark.seeds.join(', ')}],
              {' '}{benchmark.ants} ants for {benchmark.ticks} ticks.
            </span>
          </>
        ) : (
          <div className="card"><span className="section-note">No benchmark results yet. Run it to compare the three policies and six ablations.</span></div>
        )}
        <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-[#9aa8bf]">Live mutes on the canvas colony</span>
        <div className="pill-row" style={{ flexWrap: 'wrap' }}>
          {ABLATIONS.map((a) => {
            const style = { '--pill-color': ABLATION_COLOR[a.id] } as CSSProperties;
            return (
              <button
                key={a.id}
                type="button"
                className="pill"
                data-active={muted[a.id]}
                aria-pressed={muted[a.id]}
                style={style}
                onClick={() => onToggleMute(a.id)}
              >
                <span className="dot" style={{ opacity: muted[a.id] ? 0.3 : 1 }} />
                {muted[a.id] ? `${a.label} muted` : a.label.slice(1)}
              </button>
            );
          })}
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="section-title font-display">Live arena policy</h3>
        <p className="section-note">50/50 modes split the world into two sealed halves with identical layouts, one colony each.</p>
        <div className="seg" role="radiogroup" aria-label="Arena policy">
          {ARENAS.map((a) => (
            <button key={a.id} type="button" role="radio" aria-checked={arena === a.id} data-active={arena === a.id} onClick={() => onArena(a.id)}>
              {a.label}
            </button>
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="section-title font-display flex items-center gap-2"><BrainIcon size={18} /> Brain weights</h3>
        <WeightHeatmap weights={weights} version={weightsVersion} />
        <span className="section-note">Left block: 19 inputs to 16 hidden units. Right block: hidden units to 9 outputs. Cyan is excitatory, crimson inhibitory.</span>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn ghost" onClick={onExport}>Export JSON</button>
          <button type="button" className="btn ghost" onClick={() => fileRef.current?.click()}>Import JSON</button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onImport(f);
              e.target.value = '';
            }}
          />
        </div>
      </section>
    </>
  );
}
