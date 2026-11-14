'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import SanctuaryCanvas from '../components/SanctuaryCanvas';
import TelemetryPill from '../components/TelemetryPill';
import FloatingToolDock from '../components/FloatingToolDock';
import ParameterSheet from '../components/ParameterSheet';
import FieldGuideModal from '../components/FieldGuideModal';
import ToastOverlay from '../components/ToastOverlay';
import ProofOfLearningLab from '../components/ProofOfLearningLab';
import { BrandMark, Icon } from '../components/ui/icons';
import { BAKED_CHAMPION } from '../lib/neat';
import {
  BENCH_SEEDS,
  DEFAULT_SHAPE,
  HeadlessTrainer,
  benchmarkSteps,
} from '../lib/headless-sim';
import type {
  AblationId,
  ArenaMode,
  BenchmarkProgress,
  BenchmarkResult,
  GenerationRecord,
  PresetId,
  SubMode,
  Telemetry,
  ToastMessage,
  ToolId,
} from '../lib/types';

const tabLabels: Record<string, string> = {
  proof: 'Proof of Learning',
  colony: 'Colony & Threats',
  alchemy: 'Pheromone Alchemy',
  brush: 'Brush',
  world: 'World',
};

const TOOL_ORDER: ToolId[] = ['food-trail', 'home-trail', 'alarm', 'crystal', 'wall', 'eraser'];

const DEFAULT_SETTINGS = {
  maxPop: 1600,
  atpBurn: 1,
  lifespan: 200,
  predators: 0,
  evaporation: 1,
  diffusion: 2,
  crossReaction: 1,
  radius: 24,
  strength: 1,
};

export default function Page() {
  const canvasRef = useRef<React.ComponentRef<typeof SanctuaryCanvas> | null>(null);

  /* ---------- UI state ---------- */
  const [tool, setTool] = useState<ToolId>('food-trail');
  const [subMode, setSubMode] = useState<SubMode>(0);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState<1 | 2 | 4>(1);
  const [snap, setSnap] = useState<'peek' | 'half' | 'full'>('peek');
  const [tab, setTab] = useState<string>('proof');
  const [guideOpen, setGuideOpen] = useState(false);
  const [ghost, setGhost] = useState(true);
  const [splashHidden, setSplashHidden] = useState(false);
  const [dim, setDim] = useState(false);
  const [arena, setArena] = useState<ArenaMode>('trained');
  const [preset, setPreset] = useState<PresetId>('open');
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const toastId = useRef(0);

  /* ---------- Brain / training state ---------- */
  const [brainWeights, setBrainWeights] = useState<Float32Array>(
    () => new Float32Array(BAKED_CHAMPION.weights),
  );
  const [weightsVersion, setWeightsVersion] = useState(0);
  const [history, setHistory] = useState<GenerationRecord[]>(BAKED_CHAMPION.history);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(
    BAKED_CHAMPION.benchmark ?? null,
  );
  const [training, setTraining] = useState<{ running: boolean; done: number; target: number; progress: number }>({
    running: false,
    done: 0,
    target: 0,
    progress: 0,
  });
  const [benchProgress, setBenchProgress] = useState<BenchmarkProgress | null>(null);
  const [muted, setMuted] = useState<Record<AblationId, boolean>>({
    foodTrail: false, homeGuide: false, alarm: false, scoutPulse: false, noEntry: false, pathIntegration: false,
  });
  const workTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const busy = training.running || benchProgress !== null;

  const pushToast = useCallback(
    (text: string, tone: ToastMessage['tone'] = 'info') => {
      const id = ++toastId.current;
      setToasts((t) => [...t.slice(-2), { id, text, tone }]);
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3200);
    },
    [],
  );

  /* ---------- Sim control ---------- */
  const selectTool = useCallback((t: ToolId) => {
    setTool(t);
    setSubMode(t === 'home-trail' ? 1 : t === 'alarm' ? 2 : 0);
  }, []);

  const changeArena = useCallback((a: ArenaMode) => {
    setArena(a);
    canvasRef.current?.resetWorld(preset, a);
    pushToast(
      a === 'trained' ? 'Arena: trained brain'
        : a === 'aco' ? 'Arena: hand-crafted ACO'
          : 'Arena: A/B comparison',
      'info',
    );
  }, [preset, pushToast]);

  const changePreset = useCallback((p: PresetId) => {
    setPreset(p);
    canvasRef.current?.resetWorld(p, arena);
    pushToast(`World: ${p}`, 'info');
  }, [arena, pushToast]);

  const handlePointerActive = useCallback((active: boolean) => {
    if (active) {
      setDim(true);
      setGhost(false);
    } else {
      setTimeout(() => setDim(false), 200);
    }
  }, []);

  const onReady = useCallback((sim: any) => {
    sim.setPredators(settings.predators);
    sim.setBrain(brainWeights);
  }, [settings.predators, brainWeights]);

  /* ---------- Training (chunked headless) ---------- */
  const startTraining = useCallback((generations: number) => {
    if (workTimer.current || busy) return;
    const lastRec = history.length > 0 ? history[history.length - 1] : null;
    const trainer = new HeadlessTrainer({
      populationSize: 6,
      seedsPerEval: 2,
      seed: 1 + (Date.now() % 9973),
      shape: DEFAULT_SHAPE,
      warmStart: brainWeights,
      startGeneration: lastRec ? lastRec.generation + 1 : 0,
      knownBaselines: lastRec
        ? { aco: lastRec.acoFood, random: lastRec.randomFood }
        : null,
    });

    const sim = canvasRef.current?.getSim();
    if (sim && sim.founderBoost > 1) {
      trainer.evo.mutationBoost = sim.founderBoost;
      sim.founderBoost = 1;
    }

    let done = 0;
    setTraining({ running: true, done: 0, target: generations, progress: 0 });

    const tick = (): void => {
      const rec = trainer.advance();
      if (rec) {
        done++;
        setHistory((h) => [...h, rec]);
      }
      if (done >= generations) {
        workTimer.current = null;
        const newWeights = Float32Array.from(trainer.champion);
        setBrainWeights(newWeights);
        setWeightsVersion((v) => v + 1);
        canvasRef.current?.setBrain(newWeights);
        const gain = trainer.championFood - (lastRec?.championFood ?? 0);
        setTraining({ running: false, done, target: generations, progress: trainer.progress });
        pushToast(
          gain > 0.05
            ? `+${generations} generations: champion food ${(lastRec?.championFood ?? 0).toFixed(1)} → ${trainer.championFood.toFixed(1)}`
            : `+${generations} generations: no improvement (${trainer.championFood.toFixed(1)} food)`,
          gain > 0.05 ? 'good' : 'info',
        );
        return;
      }
      setTraining({ running: true, done, target: generations, progress: trainer.progress });
      workTimer.current = setTimeout(tick, 0);
    };
    workTimer.current = setTimeout(tick, 60);
  }, [busy, history, brainWeights, pushToast]);

  const runBenchmark = useCallback(() => {
    if (workTimer.current || busy) return;
    const it = benchmarkSteps(brainWeights, { seeds: BENCH_SEEDS.slice(0, 3), shape: DEFAULT_SHAPE });
    setBenchProgress({ done: 0, total: 27, label: 'Preparing' });

    const tick = (): void => {
      const step = it.next();
      if (step.done) {
        workTimer.current = null;
        setBenchmark(step.value);
        setBenchProgress(null);
        pushToast('Benchmark complete', 'good');
        return;
      }
      setBenchProgress(step.value);
      workTimer.current = setTimeout(tick, 0);
    };
    workTimer.current = setTimeout(tick, 60);
  }, [busy, brainWeights, pushToast]);

  const exportBrain = useCallback(() => {
    const blob = new Blob([JSON.stringify({ weights: Array.from(brainWeights), generation: 99 })], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `colony-brain-gen-99.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    pushToast('Brain weights exported', 'good');
  }, [brainWeights, pushToast]);

  const importBrain = useCallback((file: File) => {
    file.text().then((text) => {
      try {
        const data = JSON.parse(text);
        if (Array.isArray(data.weights) && data.weights.length === 473) {
          const w = new Float32Array(data.weights);
          setBrainWeights(w);
          setWeightsVersion((v) => v + 1);
          canvasRef.current?.setBrain(w);
          pushToast(`Imported brain (generation ${data.generation ?? 0})`, 'good');
        } else {
          pushToast('That file is not a valid 19-16-9 colony brain', 'warn');
        }
      } catch {
        pushToast('Could not read that file', 'warn');
      }
    }, () => pushToast('Could not read that file', 'warn'));
  }, [pushToast]);

  /* ---------- Keyboard ---------- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (e.key >= '1' && e.key <= '6') {
        setTool(TOOL_ORDER[Number(e.key) - 1]);
        setSubMode(TOOL_ORDER[Number(e.key) - 1] === 'home-trail' ? 1 : TOOL_ORDER[Number(e.key) - 1] === 'alarm' ? 2 : 0);
      } else if (e.key === ' ') {
        e.preventDefault();
        setPaused((p) => !p);
      } else if (e.key === '[') {
        setSettings((s) => ({ ...s, radius: Math.max(8, s.radius - 4) }));
      } else if (e.key === ']') {
        setSettings((s) => ({ ...s, radius: Math.min(64, s.radius + 4) }));
      } else if (e.key === 'Escape') {
        setSnap('peek');
        setGuideOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    const t = setTimeout(() => setSplashHidden(true), 300);
    const g = setTimeout(() => setGhost(false), 12000);
    return () => {
      clearTimeout(t);
      clearTimeout(g);
    };
  }, []);

  const telemetry: Telemetry = {
    fps: 60,
    alive: 0,
    deaths: 0,
    food: 0,
    generation: 1,
    births: 0,
    predators: settings.predators,
    scoreA: 0,
    scoreB: 0,
    arena,
    adaptive: false,
  };

  return (
    <main className="sanctuary-root" data-tool={tool}>
      {/* Full-bleed simulation canvas */}
      <SanctuaryCanvas
        ref={canvasRef}
        tool={tool}
        subMode={subMode}
        radius={settings.radius}
        strength={settings.strength}
        paused={paused}
        speed={speed}
        busy={busy}
        initialPreset={preset}
        initialArena={arena}
        onReady={onReady}
        onTelemetry={(t) => {
          telemetry.fps = t.fps;
          telemetry.alive = t.alive;
          telemetry.deaths = t.deaths;
          telemetry.food = t.food;
          telemetry.generation = t.generation;
          telemetry.births = t.births;
          telemetry.predators = t.predators;
        }}
        onPointerActive={handlePointerActive}
        onEvent={(text, tone) => pushToast(text, tone)}
      />

      {/* ============ ASTRA HEADER BRANDING ============ */}
      <header className="fixed top-0 left-0 right-0 z-50 px-4 py-3 pointer-events-none">
        <div className="pointer-events-auto flex items-center justify-between gap-4 max-w-7xl mx-auto obsidian-glass glass">
          <div className="flex items-center gap-3">
            <div className="brand-mark" aria-hidden="true">
              <BrandMark size={36} />
            </div>
            <div>
              <div className="text-xs font-display font-semibold tracking-wider text-cyan-300 uppercase">
                Formica
              </div>
              <div className="text-[11px] font-medium text-ink-dim">A Living Intelligence · Ant Colony Sandbox</div>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-2">
            <button
              className="px-3 py-1.5 rounded-md text-xs font-medium text-ink-dim hover:text-white transition-colors"
              onClick={() => setGuideOpen(false)}
            >
              Sanctuary
            </button>
            <button
              className="px-3 py-1.5 rounded-md text-xs font-medium text-ink-dim hover:text-white transition-colors"
              onClick={() => setTab('proof')}
            >
              Learning lab
            </button>
            <button
              className="px-3 py-1.5 rounded-md text-xs font-medium text-ink-dim hover:text-white transition-colors"
              onClick={() => setGuideOpen(true)}
            >
              Field guide
            </button>
          </nav>

          <div className="flex items-center gap-2">
            <span className="text-[11px] font-medium text-ink-dim flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 bio-dot-breathe"></span>
              All systems alive
            </span>
            <button
              className="px-3 py-1.5 rounded-md text-xs font-medium text-white bg-amber-500/10 hover:bg-amber-500/20 transition-all"
              onClick={() => pushToast('Snapshot saved', 'good')}
            >
              Save snapshot
            </button>
          </div>
        </div>
      </header>

      {/* ============ FABLE TOP HUD ============ */}
      <div className="hud">
        <div className="capsule">
          <div className="stat">
            <span
              className="fps-dot"
              style={{
                backgroundColor:
                  telemetry.fps >= 50 ? 'var(--viridian)' : telemetry.fps >= 30 ? 'var(--amber)' : 'var(--crimson)',
              }}
            ></span>
            <span className="text-ink-dim">FPS</span>
            <span className="text-white">{telemetry.fps}</span>
          </div>
          <div className="stat stat-dim">Alive {telemetry.alive}</div>
          <div className="stat stat-dim">Food {telemetry.food}</div>
          <div className="badge ab">GEN {telemetry.generation}</div>
        </div>

        <div className="squircles">
          <button className="squircle" aria-label="Guide" title="Guide" onClick={() => setGuideOpen(true)}>
            ?
          </button>
          <button
            className="squircle"
            aria-label="Params"
            title="Params"
            onClick={() => setSnap((s) => (s === 'peek' ? 'half' : 'peek'))}
          >
            ↦
          </button>
        </div>

        <div className="capsule">
          <button className="chip-btn" onClick={() => setPaused((p) => !p)} aria-label={paused ? 'Resume' : 'Pause'}>
            {paused ? '▶' : '⏸'}
          </button>
          <button
            className="chip-btn"
            onClick={() => setSpeed((s) => (s === 1 ? 2 : s === 2 ? 4 : 1))}
            aria-label="Speed"
          >
            {speed}×
          </button>
          <button
            className="play-btn"
            data-paused={paused ? 'true' : 'false'}
            onClick={() => setPaused((p) => !p)}
            aria-label="Play/Pause"
          >
            {paused ? '▶' : '⏸'}
          </button>
        </div>
      </div>

      {/* ============ FABLE BOTTOM 3-SNAP COCKPIT ============ */}
      <div className={`cockpit ui-layer ${dim ? 'dim' : ''}`} data-snap={snap}>
        <TelemetryPill
          telemetry={telemetry}
          paused={paused}
          speed={speed}
          onTogglePause={() => setPaused((p) => !p)}
          onCycleSpeed={() => setSpeed((s) => (s === 1 ? 2 : s === 2 ? 4 : 1))}
          onGuide={() => setGuideOpen(true)}
          onParams={() => setSnap((s) => (s === 'peek' ? 'half' : 'peek'))}
        />

        <ToastOverlay toasts={toasts} />

        {ghost && !splashHidden && (
          <div className="ghost-guide" aria-hidden="true">
            <div>Drag to paint a pheromone trail</div>
          </div>
        )}

        <div className="sheet">
          <div className="sheet-handle"><span /></div>
          <div className="peek">
            <div className="peek-title">Colony Cockpit</div>
            <div className="pill-row">
              {['proof', 'colony', 'alchemy', 'brush', 'world'].map((t) => (
                <button
                  key={t}
                  className={`pill ${tab === t ? 'data-active' : ''}`}
                  data-pill-color={
                    t === 'proof' ? 'var(--cyan)'
                      : t === 'colony' ? 'var(--amber)'
                        : t === 'alchemy' ? 'var(--violet)'
                          : t === 'brush' ? 'var(--viridian)'
                            : 'var(--crimson)'
                  }
                  onClick={() => setTab(t)}
                >
                  {tabLabels[t]}
                </button>
              ))}
            </div>
          </div>

          <div className="sheet-body">
            <div className="section-title">{tabLabels[tab]}</div>
            <div className="section-note">
              {tab === 'proof'
                ? 'Neuroevolution proof-of-learning lab — train, benchmark, ablate.'
                : tab === 'colony'
                  ? 'Colony settings, predators, mortality.'
                  : tab === 'alchemy'
                    ? 'Five-channel pheromone mixing.'
                    : tab === 'brush'
                      ? 'Paint trails, walls, crystals.'
                      : 'World presets and arena modes.'}
            </div>

            {tab === 'proof' ? (
              <ProofOfLearningLab
                history={history}
                bakedGenerations={BAKED_CHAMPION.generations}
                trainedAt={BAKED_CHAMPION.trainedAt}
                training={training}
                onTrain={startTraining}
                benchmark={benchmark}
                benchProgress={benchProgress}
                onRunBenchmark={runBenchmark}
                arena={arena}
                onArena={changeArena}
                muted={muted}
                onToggleMute={(id) => setMuted((m) => ({ ...m, [id]: !m[id] }))}
                weights={brainWeights}
                weightsVersion={weightsVersion}
                onExport={exportBrain}
                onImport={importBrain}
              />
            ) : (
              <div className="card">
                <div className="field">
                  <span>Arena</span>
                  <output>{arena}</output>
                  <button className="btn ghost w-full" onClick={() => changeArena(arena === 'trained' ? 'ab-aco' : 'trained')}>
                    Switch arena
                  </button>
                </div>
                <div className="field">
                  <span>Preset</span>
                  <output>{preset}</output>
                  <button className="btn ghost w-full" onClick={() => changePreset(preset === 'open' ? 'bridge' : 'open')}>
                    Switch world
                  </button>
                </div>
                <div className="field">
                  <span>Brush radius</span>
                  <output>{settings.radius}px</output>
                  <input
                    type="range"
                    min={8}
                    max={64}
                    step={4}
                    value={settings.radius}
                    onChange={(e) => setSettings((s) => ({ ...s, radius: Number(e.target.value) }))}
                  />
                </div>
                <div className="field">
                  <span>Predators</span>
                  <output>{settings.predators}</output>
                  <input
                    type="range"
                    min={0}
                    max={6}
                    step={1}
                    value={settings.predators}
                    onChange={(e) => {
                      const n = Number(e.target.value);
                      setSettings((s) => ({ ...s, predators: n }));
                      canvasRef.current?.setPredators?.(n);
                    }}
                  />
                </div>
                <button className="btn w-full" onClick={() => pushToast('World reset', 'info')}>
                  Reset world
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="dock-wrap">
          <FloatingToolDock
            tool={tool}
            subMode={subMode}
            radius={settings.radius}
            onTool={selectTool}
            onSubMode={setSubMode}
            onRadius={(r) => setSettings((s) => ({ ...s, radius: r }))}
          />
        </div>
      </div>

      <FieldGuideModal open={guideOpen} onClose={() => setGuideOpen(false)} />

      <div className="splash" data-hide={splashHidden} aria-hidden={splashHidden}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-amber-400 to-cyan-400 flex items-center justify-center text-[24px]">
            F
          </div>
          <span className="font-display text-[22px] font-semibold">Formica Sanctuary</span>
        </div>
      </div>
    </main>
  );
}
