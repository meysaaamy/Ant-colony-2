/**
 * Fast headless evaluator (no DOM): deterministic training episodes, the chunkable neuroevolution trainer,
 * the Food/Gen curve against Hand-Crafted ACO + Random floors, and the multi-seed A/B + ablation benchmark.
 */
import { ColonySim, FIXED_DT } from './ant-soa-engine';
import { DEFAULT_EVO, GENOME_SIZE, NeuroEvolution, createRandomGenome } from './neat';
import { applyPreset } from './presets';
import { Rng } from './prng-noise';
import { CHANNEL_KEYS, POLICY_ACO, POLICY_RANDOM, POLICY_TRAINED } from './types';
import type {
  AblationId, BenchmarkProgress, BenchmarkResult, BenchmarkRow, EpisodeConfig, EpisodeResult, GenerationRecord,
} from './types';

export const HEADLESS_WORLD = 480;
export const HEADLESS_ANTS = 72;
export const HEADLESS_TICKS = 1200;
/** Fixed seeds for the learning curve (one per scenario family). */
export const CURVE_SEEDS: readonly number[] = [101, 202, 303];
/** Held-out seeds: never used for selection during training. */
export const BENCH_SEEDS: readonly number[] = [9001, 9002, 9003, 9004, 9005];

export const ABLATIONS: ReadonlyArray<{ id: AblationId; label: string }> = [
  { id: 'foodTrail', label: '-foodTrail' },
  { id: 'homeGuide', label: '-homeGuide' },
  { id: 'alarm', label: '-alarm' },
  { id: 'scoutPulse', label: '-scoutPulse' },
  { id: 'noEntry', label: '-noEntry' },
  { id: 'pathIntegration', label: '-PathIntegration' },
];

let pooled: ColonySim | null = null;

function getSim(size: number): ColonySim {
  if (!pooled || pooled.width !== size) pooled = new ColonySim(size, size, 1);
  return pooled;
}

/** Scenario family by seed: 0 and 2 -> Open Sanctuary (random caches + rocks), 1 -> Double Bridge. */
export function scenarioForSeed(seed: number): 'open' | 'bridge' {
  return seed % 3 === 1 ? 'bridge' : 'open';
}

export function applyAblation(sim: ColonySim, ablation: AblationId | null): void {
  sim.grid.mask.fill(1);
  sim.pathIntegration = true;
  if (ablation === null) return;
  if (ablation === 'pathIntegration') {
    sim.pathIntegration = false;
    return;
  }
  const ch = CHANNEL_KEYS.indexOf(ablation);
  if (ch >= 0) sim.grid.mask[ch] = 0;
}

/** One deterministic episode. Births and founder waves are off so every policy keeps the same headcount. */
export function runEpisode(cfg: EpisodeConfig): EpisodeResult {
  const sim = getSim(cfg.worldSize);
  sim.rng.reseed(cfg.seed * 2654435761 + 12345);
  sim.params.maxPop = cfg.ants;
  sim.params.atpBurn = 2;
  sim.params.lifespan = 600;
  sim.params.mortality = true;
  sim.params.births = false;
  sim.params.founderWaves = false;
  sim.halfRateDiffusion = true;
  sim.grid.evaporation = 1;
  sim.grid.diffusion = 2;
  sim.grid.crossReaction = 1;
  sim.arena = 'trained';
  sim.policyA = cfg.policy;
  sim.policyB = POLICY_RANDOM;
  if (cfg.policy === POLICY_TRAINED && cfg.weights) sim.brainA.set(cfg.weights);
  if (cfg.policy === POLICY_RANDOM) createRandomGenome(new Rng(cfg.seed * 7919 + 13), DEFAULT_EVO.initScale, sim.brainB);
  applyAblation(sim, cfg.ablation);
  applyPreset(sim, scenarioForSeed(cfg.seed), cfg.seed, cfg.ants);
  for (let t = 0; t < cfg.ticks; t++) sim.step(FIXED_DT);
  const food = sim.delivered[0];
  const result: EpisodeResult = {
    food,
    pickups: sim.pickups,
    fitness: sim.colonyFitness(),
    efficiency: food > 0 ? sim.effSum / food : 0,
    deaths: sim.totalDeaths,
    wallRate: sim.wallHits / Math.max(1, sim.antTicks),
    loops: sim.loops,
  };
  applyAblation(sim, null);
  return result;
}

export interface EvalSummary {
  fitness: number;
  food: number;
}

export interface EvalShape {
  ants: number;
  ticks: number;
  worldSize: number;
}

export const DEFAULT_SHAPE: EvalShape = { ants: HEADLESS_ANTS, ticks: HEADLESS_TICKS, worldSize: HEADLESS_WORLD };

export function evaluateWeights(weights: Float32Array, seeds: readonly number[], shape: EvalShape): EvalSummary {
  let fitness = 0;
  let food = 0;
  for (let s = 0; s < seeds.length; s++) {
    const r = runEpisode({
      seed: seeds[s], ants: shape.ants, ticks: shape.ticks, worldSize: shape.worldSize,
      policy: POLICY_TRAINED, weights, ablation: null,
    });
    fitness += r.fitness;
    food += r.food;
  }
  return { fitness: fitness / seeds.length, food: food / seeds.length };
}

export function evaluateBaseline(policy: 1 | 2, seeds: readonly number[], shape: EvalShape): number {
  let food = 0;
  for (let s = 0; s < seeds.length; s++) {
    food += runEpisode({
      seed: seeds[s], ants: shape.ants, ticks: shape.ticks, worldSize: shape.worldSize,
      policy, weights: null, ablation: null,
    }).food;
  }
  return food / seeds.length;
}

/* ------------------------------------------------------------------ */
/* Trainer                                                             */
/* ------------------------------------------------------------------ */

export interface TrainerOptions {
  populationSize: number;
  seedsPerEval: number;
  shape: EvalShape;
  seed: number;
  speciation: boolean;
  warmStart: Float32Array | null;
  /** Generation index to start counting from (lets the UI continue a baked curve). */
  startGeneration: number;
  /** Skip recomputing the ACO / Random floors when they are already known for this shape. */
  knownBaselines: { aco: number; random: number } | null;
}

export const DEFAULT_TRAINER: TrainerOptions = {
  populationSize: 24,
  seedsPerEval: 3,
  shape: DEFAULT_SHAPE,
  seed: 1,
  speciation: false,
  warmStart: null,
  startGeneration: 0,
  knownBaselines: null,
};

/**
 * Neuroevolution over candidate colony brains.
 *  - fitness: mean colony fitness over `seedsPerEval` seeds that rotate every generation (anti-overfit)
 *  - curve: each generation's best genome is re-scored on the fixed CURVE_SEEDS (N = 3) for Food/Gen
 *  - champion: best genome ever by curve-seed fitness
 * `advance()` runs exactly one episode so browsers can slice the work across frames (chunked trainer).
 */
export class HeadlessTrainer {
  readonly opts: TrainerOptions;
  readonly evo: NeuroEvolution;
  readonly history: GenerationRecord[] = [];
  readonly champion = new Float32Array(GENOME_SIZE);
  championFitness = -Infinity;
  championFood = 0;
  acoFood = 0;
  randomFood = 0;
  generation: number;
  private baselinesReady = false;
  private cursor = 0;
  private seedCursor = 0;
  private fitAcc = 0;
  private phase = 0;
  private bestIdx = 0;
  private valIdx = 0;
  private valFit = 0;
  private valFood = 0;
  private episodesDone = 0;
  private readonly genSeeds: number[] = [];

  constructor(options: Partial<TrainerOptions>) {
    this.opts = { ...DEFAULT_TRAINER, ...options };
    this.evo = new NeuroEvolution(
      { populationSize: this.opts.populationSize, seed: this.opts.seed, speciation: this.opts.speciation },
      this.opts.warmStart,
    );
    this.generation = this.opts.startGeneration;
    if (this.opts.warmStart) this.champion.set(this.opts.warmStart);
    else this.champion.set(this.evo.population[0]);
    this.rollSeeds();
  }

  private rollSeeds(): void {
    this.genSeeds.length = 0;
    const spe = this.opts.seedsPerEval;
    // consecutive integers walk through every scenario family (seed % 3); the stride keeps them rotating
    const stride = spe % 3 === 0 ? 3 : spe;
    const base = 1000 + this.opts.seed * 97 + this.generation * stride;
    for (let k = 0; k < spe; k++) this.genSeeds.push(base + k);
  }

  get episodesPerGeneration(): number {
    return this.opts.populationSize * this.opts.seedsPerEval + CURVE_SEEDS.length;
  }

  /** Fraction of the current generation already evaluated. */
  get progress(): number {
    return this.episodesDone / this.episodesPerGeneration;
  }

  /** ACO + Random floors on the curve seeds, and the warm-start champion's own score. */
  computeBaselines(): void {
    if (this.baselinesReady) return;
    if (this.opts.knownBaselines) {
      this.acoFood = this.opts.knownBaselines.aco;
      this.randomFood = this.opts.knownBaselines.random;
    } else {
      this.acoFood = evaluateBaseline(POLICY_ACO, CURVE_SEEDS, this.opts.shape);
      this.randomFood = evaluateBaseline(POLICY_RANDOM, CURVE_SEEDS, this.opts.shape);
    }
    if (this.opts.warmStart) {
      const s = evaluateWeights(this.champion, CURVE_SEEDS, this.opts.shape);
      this.championFitness = s.fitness;
      this.championFood = s.food;
    }
    this.baselinesReady = true;
  }

  private episode(weights: Float32Array, seed: number): EpisodeResult {
    const shape = this.opts.shape;
    return runEpisode({
      seed, ants: shape.ants, ticks: shape.ticks, worldSize: shape.worldSize,
      policy: POLICY_TRAINED, weights, ablation: null,
    });
  }

  /** Runs one episode. Returns the generation record when a generation just completed, else null. */
  advance(): GenerationRecord | null {
    this.computeBaselines();
    const evo = this.evo;
    this.episodesDone++;
    if (this.phase === 0) {
      const i = this.cursor;
      this.fitAcc += this.episode(evo.population[i], this.genSeeds[this.seedCursor]).fitness;
      this.seedCursor++;
      if (this.seedCursor >= this.genSeeds.length) {
        evo.fitness[i] = this.fitAcc / this.genSeeds.length;
        this.fitAcc = 0;
        this.seedCursor = 0;
        this.cursor++;
        if (this.cursor >= this.opts.populationSize) {
          this.phase = 1;
          this.bestIdx = evo.bestIndex();
          this.valIdx = 0;
          this.valFit = 0;
          this.valFood = 0;
        }
      }
      return null;
    }
    // phase 1: re-score this generation's best genome on the fixed curve seeds
    const r = this.episode(evo.population[this.bestIdx], CURVE_SEEDS[this.valIdx]);
    this.valFit += r.fitness;
    this.valFood += r.food;
    this.valIdx++;
    if (this.valIdx < CURVE_SEEDS.length) return null;

    const valFitness = this.valFit / CURVE_SEEDS.length;
    const valFood = this.valFood / CURVE_SEEDS.length;
    const improved = valFitness > this.championFitness;
    if (improved) {
      this.championFitness = valFitness;
      this.championFood = valFood;
      this.champion.set(evo.population[this.bestIdx]);
    }
    const record: GenerationRecord = {
      generation: this.generation,
      bestFood: valFood,
      championFood: this.championFood,
      bestFitness: evo.fitness[this.bestIdx],
      meanFitness: evo.meanFitness(),
      acoFood: this.acoFood,
      randomFood: this.randomFood,
      sigma: evo.sigma,
      species: evo.speciesCount,
    };
    this.history.push(record);
    evo.noteProgress(improved);
    evo.evolve();
    this.generation++;
    this.cursor = 0;
    this.phase = 0;
    this.episodesDone = 0;
    this.rollSeeds();
    return record;
  }

  runGeneration(): GenerationRecord {
    let rec: GenerationRecord | null = null;
    while (rec === null) rec = this.advance();
    return rec;
  }
}

/* ------------------------------------------------------------------ */
/* Multi-seed A/B + ablation benchmark                                 */
/* ------------------------------------------------------------------ */

export interface BenchmarkOptions {
  seeds: readonly number[];
  shape: EvalShape;
}

function summarize(id: string, label: string, kind: 'policy' | 'ablation', samples: number[]): BenchmarkRow {
  let mean = 0;
  for (const s of samples) mean += s;
  mean /= samples.length;
  let v = 0;
  for (const s of samples) v += (s - mean) * (s - mean);
  const std = samples.length > 1 ? Math.sqrt(v / (samples.length - 1)) : 0;
  return { id, label, kind, mean, std, samples, delta: 0 };
}

/**
 * Generator form: yields after every episode so the UI can run it in slices.
 * Rows: Full Trained Brain, Hand-Crafted ACO, Random Brain, then six single-input ablations.
 */
export function* benchmarkSteps(
  weights: Float32Array,
  opts: BenchmarkOptions,
): Generator<BenchmarkProgress, BenchmarkResult, void> {
  const { seeds, shape } = opts;
  const rows: BenchmarkRow[] = [];
  const total = (3 + ABLATIONS.length) * seeds.length;
  let done = 0;
  const run = (policy: 0 | 1 | 2, ablation: AblationId | null, seed: number): number =>
    runEpisode({
      seed, ants: shape.ants, ticks: shape.ticks, worldSize: shape.worldSize,
      policy, weights: policy === POLICY_TRAINED ? weights : null, ablation,
    }).food;

  const plan: Array<{ id: string; label: string; kind: 'policy' | 'ablation'; policy: 0 | 1 | 2; ablation: AblationId | null }> = [
    { id: 'trained', label: 'Full Trained Brain', kind: 'policy', policy: POLICY_TRAINED, ablation: null },
    { id: 'aco', label: 'Hand-Crafted ACO', kind: 'policy', policy: POLICY_ACO, ablation: null },
    { id: 'random', label: 'Random Brain', kind: 'policy', policy: POLICY_RANDOM, ablation: null },
  ];
  for (const a of ABLATIONS) plan.push({ id: a.id, label: a.label, kind: 'ablation', policy: POLICY_TRAINED, ablation: a.id });

  for (const p of plan) {
    const samples: number[] = [];
    for (const seed of seeds) {
      samples.push(run(p.policy, p.ablation, seed));
      done++;
      yield { done, total, label: p.label };
    }
    rows.push(summarize(p.id, p.label, p.kind, samples));
  }
  const full = rows[0].mean;
  for (const r of rows) r.delta = r.mean - full;
  return { seeds: [...seeds], ants: shape.ants, ticks: shape.ticks, rows };
}

export function runBenchmark(weights: Float32Array, opts: BenchmarkOptions): BenchmarkResult {
  const it = benchmarkSteps(weights, opts);
  let step = it.next();
  while (!step.done) step = it.next();
  return step.value;
}

export interface OrderingVerdict {
  trained: number;
  aco: number;
  random: number;
  trainedBeatsAco: boolean;
  acoBeatsRandom: boolean;
  /** Ablations whose mean food fell below the full brain. */
  drops: string[];
}

export function verifyOrdering(result: BenchmarkResult): OrderingVerdict {
  const get = (id: string): number => {
    for (const r of result.rows) if (r.id === id) return r.mean;
    return 0;
  };
  const trained = get('trained');
  const aco = get('aco');
  const random = get('random');
  const drops: string[] = [];
  for (const r of result.rows) if (r.kind === 'ablation' && r.delta < 0) drops.push(r.label);
  return { trained, aco, random, trainedBeatsAco: trained > aco, acoBeatsRandom: aco > random, drops };
}
