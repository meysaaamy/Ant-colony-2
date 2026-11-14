/**
 * Unified Shared Colony Brain.
 *
 * Fixed-topology recurrent MLP: 19 inputs -> 16 hidden (tanh) -> 9 outputs.
 * Recurrence is Jordan-style: outputs 7 and 8 (nextMemA, nextMemB) are fed back as inputs 17 and 18.
 * One flat Float32Array genome (473 weights) is shared by every TRAINED_MLP ant.
 *
 * Neuroevolution: elitism + tournament selection + neuron-wise crossover + adaptive Gaussian mutation,
 * with optional NEAT-style speciation (compatibility distance + fitness sharing). Topology is fixed.
 */
import type { BakedChampion, BrainJSON } from './types';
import { Rng } from './prng-noise';

export const BRAIN_INPUTS = 19;
export const BRAIN_HIDDEN = 16;
export const BRAIN_OUTPUTS = 9;
export const OFF_B1 = BRAIN_INPUTS * BRAIN_HIDDEN; // 304
export const OFF_W2 = OFF_B1 + BRAIN_HIDDEN; // 320
export const OFF_B2 = OFF_W2 + BRAIN_HIDDEN * BRAIN_OUTPUTS; // 464
export const GENOME_SIZE = OFF_B2 + BRAIN_OUTPUTS; // 473
export const WEIGHT_LIMIT = 8;

export const INPUT_LABELS: readonly string[] = [
  'grad foodTrail', 'grad homeGuide', 'grad alarm', 'grad scoutPulse', 'grad noEntry',
  'ctr foodTrail', 'ctr homeGuide', 'ctr alarm', 'ctr scoutPulse', 'ctr noEntry',
  'wall L', 'wall C', 'wall R',
  'home fwd', 'home lat',
  'carrying', 'energy', 'memA', 'memB',
];

export const OUTPUT_LABELS: readonly string[] = [
  'steerTorque', 'speedMod', 'emitFood', 'emitHome', 'emitAlarm', 'emitScout', 'emitNoEntry', 'nextMemA', 'nextMemB',
];

/**
 * Forward pass. out[0] steer in [-1,1] (tanh), out[1..6] in [0,1] (sigmoid), out[7..8] in [-1,1] (tanh).
 * Allocation free: caller owns all buffers.
 */
export function forward(w: Float32Array, inp: Float32Array, hid: Float32Array, out: Float32Array): void {
  let k = 0;
  for (let j = 0; j < BRAIN_HIDDEN; j++) {
    let s = w[OFF_B1 + j];
    for (let i = 0; i < BRAIN_INPUTS; i++) s += w[k++] * inp[i];
    hid[j] = Math.tanh(s);
  }
  k = OFF_W2;
  for (let o = 0; o < BRAIN_OUTPUTS; o++) {
    let s = w[OFF_B2 + o];
    for (let j = 0; j < BRAIN_HIDDEN; j++) s += w[k++] * hid[j];
    out[o] = o === 0 || o >= 7 ? Math.tanh(s) : 1 / (1 + Math.exp(-s));
  }
}

export function createRandomGenome(rng: Rng, scale: number, target?: Float32Array): Float32Array {
  const g = target ?? new Float32Array(GENOME_SIZE);
  for (let i = 0; i < GENOME_SIZE; i++) g[i] = rng.gauss() * scale;
  return g;
}

export function genomeDistance(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < GENOME_SIZE; i++) s += Math.abs(a[i] - b[i]);
  return s / GENOME_SIZE;
}

/* ------------------------------------------------------------------ */
/* Neuroevolution engine                                               */
/* ------------------------------------------------------------------ */

export interface EvoOptions {
  populationSize: number;
  elites: number;
  tournament: number;
  crossoverRate: number;
  mutationRate: number;
  sigma: number;
  sigmaMin: number;
  sigmaMax: number;
  initScale: number;
  speciation: boolean;
  compatThreshold: number;
  seed: number;
}

export const DEFAULT_EVO: EvoOptions = {
  populationSize: 24,
  elites: 3,
  tournament: 3,
  crossoverRate: 0.55,
  mutationRate: 0.1,
  sigma: 0.4,
  sigmaMin: 0.05,
  sigmaMax: 0.5,
  initScale: 0.7,
  speciation: false,
  compatThreshold: 0.45,
  seed: 1,
};

export class NeuroEvolution {
  readonly opts: EvoOptions;
  population: Float32Array[] = [];
  private scratch: Float32Array[] = [];
  readonly fitness: Float32Array;
  readonly species: Int32Array;
  private readonly adjusted: Float32Array;
  private readonly order: Int32Array;
  private readonly speciesSize: Int32Array;
  private readonly rng: Rng;
  generation = 0;
  sigma: number;
  /** Temporary multiplier (e.g. 2.5x Founder Bottleneck boost). Resets after one evolve(). */
  mutationBoost = 1;
  stagnation = 0;
  speciesCount = 1;

  constructor(options: Partial<EvoOptions>, warmStart?: Float32Array | null) {
    this.opts = { ...DEFAULT_EVO, ...options };
    const n = this.opts.populationSize;
    this.rng = new Rng(this.opts.seed);
    // warm starts refine an existing champion: begin with small steps
    this.sigma = warmStart && warmStart.length === GENOME_SIZE ? Math.min(this.opts.sigma, 0.16) : this.opts.sigma;
    this.fitness = new Float32Array(n);
    this.adjusted = new Float32Array(n);
    this.species = new Int32Array(n);
    this.speciesSize = new Int32Array(n);
    this.order = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const g = new Float32Array(GENOME_SIZE);
      if (warmStart && warmStart.length === GENOME_SIZE) {
        g.set(warmStart);
        if (i > 0) this.mutate(g, i < n / 2 ? 0.5 : 1);
      } else {
        createRandomGenome(this.rng, this.opts.initScale, g);
      }
      this.population.push(g);
      this.scratch.push(new Float32Array(GENOME_SIZE));
    }
  }

  bestIndex(): number {
    let b = 0;
    for (let i = 1; i < this.fitness.length; i++) if (this.fitness[i] > this.fitness[b]) b = i;
    return b;
  }

  meanFitness(): number {
    let s = 0;
    for (let i = 0; i < this.fitness.length; i++) s += this.fitness[i];
    return s / this.fitness.length;
  }

  /** Adaptive mutation: shrink sigma on progress, widen it when the champion stagnates. */
  noteProgress(improved: boolean): void {
    if (improved) {
      this.stagnation = 0;
      this.sigma = Math.max(this.opts.sigmaMin, this.sigma * 0.92);
    } else {
      this.stagnation++;
      if (this.stagnation >= 3) this.sigma = Math.min(this.opts.sigmaMax, this.sigma * 1.18);
    }
  }

  private mutate(g: Float32Array, strength: number): void {
    const rng = this.rng;
    const rate = this.opts.mutationRate;
    const sd = this.sigma * strength * this.mutationBoost;
    for (let i = 0; i < GENOME_SIZE; i++) {
      const r = rng.next();
      if (r < rate) {
        let v = g[i] + rng.gauss() * sd;
        if (r < rate * 0.04) v = rng.gauss() * this.opts.initScale;
        g[i] = v > WEIGHT_LIMIT ? WEIGHT_LIMIT : v < -WEIGHT_LIMIT ? -WEIGHT_LIMIT : v;
      }
    }
  }

  /** NEAT-style speciation by mean absolute weight distance to species representatives. */
  private speciate(): void {
    const n = this.population.length;
    let count = 0;
    this.speciesSize.fill(0);
    for (let i = 0; i < n; i++) {
      let assigned = -1;
      for (let j = 0; j < i; j++) {
        // the first member of each species acts as its representative
        if (this.species[j] !== -1 && this.isRepresentative(j) &&
          genomeDistance(this.population[i], this.population[j]) < this.opts.compatThreshold) {
          assigned = this.species[j];
          break;
        }
      }
      if (assigned < 0) assigned = count++;
      this.species[i] = assigned;
      this.speciesSize[assigned]++;
    }
    this.speciesCount = count;
  }

  private isRepresentative(j: number): boolean {
    const s = this.species[j];
    for (let k = 0; k < j; k++) if (this.species[k] === s) return false;
    return true;
  }

  private tournament(sameSpeciesAs: number): number {
    const n = this.population.length;
    let best = -1;
    for (let t = 0; t < this.opts.tournament; t++) {
      let c = this.rng.int(n);
      if (sameSpeciesAs >= 0) {
        // a few retries to stay within the species (NEAT mating restriction)
        for (let tries = 0; tries < 4 && this.species[c] !== this.species[sameSpeciesAs]; tries++) c = this.rng.int(n);
      }
      if (best < 0 || this.adjusted[c] > this.adjusted[best]) best = c;
    }
    return best;
  }

  /** Neuron-wise uniform crossover keeps each hidden unit's fan-in and fan-out together. */
  private crossover(a: Float32Array, b: Float32Array, child: Float32Array): void {
    child.set(a);
    for (let j = 0; j < BRAIN_HIDDEN; j++) {
      if (this.rng.next() < 0.5) continue;
      const base = j * BRAIN_INPUTS;
      for (let i = 0; i < BRAIN_INPUTS; i++) child[base + i] = b[base + i];
      child[OFF_B1 + j] = b[OFF_B1 + j];
      for (let o = 0; o < BRAIN_OUTPUTS; o++) child[OFF_W2 + o * BRAIN_HIDDEN + j] = b[OFF_W2 + o * BRAIN_HIDDEN + j];
    }
    for (let o = 0; o < BRAIN_OUTPUTS; o++) if (this.rng.next() < 0.5) child[OFF_B2 + o] = b[OFF_B2 + o];
  }

  /** Consumes this.fitness and produces the next generation in place. */
  evolve(): void {
    const n = this.population.length;
    const fit = this.fitness;
    if (this.opts.speciation) {
      this.speciate();
      let min = Infinity;
      for (let i = 0; i < n; i++) if (fit[i] < min) min = fit[i];
      for (let i = 0; i < n; i++) this.adjusted[i] = (fit[i] - min + 1e-3) / this.speciesSize[this.species[i]];
    } else {
      this.species.fill(0);
      this.speciesCount = 1;
      for (let i = 0; i < n; i++) this.adjusted[i] = fit[i];
    }
    // rank by raw fitness for elitism (insertion sort on a tiny Int32Array)
    for (let i = 0; i < n; i++) this.order[i] = i;
    for (let i = 1; i < n; i++) {
      const v = this.order[i];
      let j = i - 1;
      while (j >= 0 && fit[this.order[j]] < fit[v]) {
        this.order[j + 1] = this.order[j];
        j--;
      }
      this.order[j + 1] = v;
    }
    const elites = Math.min(this.opts.elites, n);
    for (let e = 0; e < elites; e++) this.scratch[e].set(this.population[this.order[e]]);
    for (let c = elites; c < n; c++) {
      const pa = this.tournament(-1);
      const child = this.scratch[c];
      if (this.rng.next() < this.opts.crossoverRate) {
        const pb = this.tournament(this.opts.speciation && this.rng.next() < 0.9 ? pa : -1);
        this.crossover(this.population[pa], this.population[pb], child);
      } else {
        child.set(this.population[pa]);
      }
      // log-uniform strength in [0.25, 2] x sigma: every brood mixes fine-tuning with exploration
      this.mutate(child, 0.25 * Math.pow(2, this.rng.next() * 3));
    }
    const t = this.population;
    this.population = this.scratch;
    this.scratch = t;
    this.mutationBoost = 1;
    this.generation++;
  }
}

/* ------------------------------------------------------------------ */
/* JSON import / export                                                */
/* ------------------------------------------------------------------ */

export function exportBrainJSON(weights: Float32Array, generation: number): BrainJSON {
  const w: number[] = new Array<number>(GENOME_SIZE);
  for (let i = 0; i < GENOME_SIZE; i++) w[i] = Math.round(weights[i] * 10000) / 10000;
  return {
    format: 'cryo-colony-brain',
    version: 1,
    inputs: BRAIN_INPUTS,
    hidden: BRAIN_HIDDEN,
    outputs: BRAIN_OUTPUTS,
    generation,
    weights: w,
  };
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

/** Validates and imports a brain. Returns null on any structural problem. */
export function importBrainJSON(raw: unknown): { weights: Float32Array; generation: number } | null {
  let data: unknown = raw;
  if (typeof raw === 'string') {
    try {
      data = JSON.parse(raw) as unknown;
    } catch {
      return null;
    }
  }
  if (!isRecord(data)) return null;
  if (data.format !== 'cryo-colony-brain') return null;
  if (data.inputs !== BRAIN_INPUTS || data.hidden !== BRAIN_HIDDEN || data.outputs !== BRAIN_OUTPUTS) return null;
  const w = data.weights;
  if (!Array.isArray(w) || w.length !== GENOME_SIZE) return null;
  const weights = new Float32Array(GENOME_SIZE);
  for (let i = 0; i < GENOME_SIZE; i++) {
    const v: unknown = w[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) return null;
    weights[i] = v > WEIGHT_LIMIT ? WEIGHT_LIMIT : v < -WEIGHT_LIMIT ? -WEIGHT_LIMIT : v;
  }
  const generation = typeof data.generation === 'number' && Number.isFinite(data.generation) ? data.generation : 0;
  return { weights, generation };
}

/* ------------------------------------------------------------------ */
/* Baked champion (rewritten by `npx tsx scripts/train.ts --bake`)      */
/* ------------------------------------------------------------------ */

const CHAMPION_BEGIN = '/* @@CHAMPION_BEGIN@@ */';
const CHAMPION_END = '/* @@CHAMPION_END@@ */';

/* @@CHAMPION_BEGIN@@ */
export const BAKED_CHAMPION: BakedChampion = {"version":1,"trainedAt":"2026-11-13","generations":95,"populationSize":16,"trainSeed":14,"history":[{"generation":0,"bestFood":6.666666666666667,"championFood":6.666666666666667,"bestFitness":19.61409568786621,"meanFitness":-3.5545946173369884,"acoFood":93,"randomFood":0,"sigma":0.4,"species":1},{"generation":1,"bestFood":2.6666666666666665,"championFood":6.666666666666667,"bestFitness":380.2808532714844,"meanFitness":77.99984957378547,"acoFood":93,"randomFood":0,"sigma":0.36800000000000005,"species":1},{"generation":2,"bestFood":10.333333333333334,"championFood":10.333333333333334,"bestFitness":227.43653869628906,"meanFitness":83.89641029760242,"acoFood":93,"randomFood":0,"sigma":0.36800000000000005,"species":1},{"generation":3,"bestFood":2.6666666666666665,"championFood":10.333333333333334,"bestFitness":232.68609619140625,"meanFitness":52.78817090205848,"acoFood":93,"randomFood":0,"sigma":0.3385600000000001,"species":1},{"generation":4,"bestFood":11.333333333333334,"championFood":11.333333333333334,"bestFitness":278.5370178222656,"meanFitness":73.05026609823108,"acoFood":93,"randomFood":0,"sigma":0.3385600000000001,"species":1},{"generation":5,"bestFood":22.666666666666668,"championFood":22.666666666666668,"bestFitness":462.1194763183594,"meanFitness":116.1970689073205,"acoFood":93,"randomFood":0,"sigma":0.31147520000000006,"species":1},{"generation":6,"bestFood":22.666666666666668,"championFood":22.666666666666668,"bestFitness":451.8479309082031,"meanFitness":109.09620970863034,"acoFood":93,"randomFood":0,"sigma":0.2865571840000001,"species":1},{"generation":7,"bestFood":8.333333333333334,"championFood":22.666666666666668,"bestFitness":945.0462646484375,"meanFitness":240.105835695751,"acoFood":93,"randomFood":0,"sigma":0.2865571840000001,"species":1},{"generation":8,"bestFood":14,"championFood":22.666666666666668,"bestFitness":745.3013305664062,"meanFitness":216.82675795257092,"acoFood":93,"randomFood":0,"sigma":0.2865571840000001,"species":1},{"generation":9,"bestFood":16.333333333333332,"championFood":22.666666666666668,"bestFitness":594.97021484375,"meanFitness":246.91884994506836,"acoFood":93,"randomFood":0,"sigma":0.3381374771200001,"species":1},{"generation":10,"bestFood":27.333333333333332,"championFood":27.333333333333332,"bestFitness":562.3279418945312,"meanFitness":241.6110364496708,"acoFood":93,"randomFood":0,"sigma":0.3990022230016001,"species":1},{"generation":11,"bestFood":16.333333333333332,"championFood":27.333333333333332,"bestFitness":607.4613647460938,"meanFitness":220.51436983048916,"acoFood":93,"randomFood":0,"sigma":0.3670820451614721,"species":1},{"generation":12,"bestFood":19,"championFood":27.333333333333332,"bestFitness":612.2431030273438,"meanFitness":275.8624225789681,"acoFood":93,"randomFood":0,"sigma":0.3670820451614721,"species":1},{"generation":13,"bestFood":3.3333333333333335,"championFood":27.333333333333332,"bestFitness":615.49462890625,"meanFitness":212.7545021213591,"acoFood":93,"randomFood":0,"sigma":0.3670820451614721,"species":1},{"generation":14,"bestFood":21.333333333333332,"championFood":27.333333333333332,"bestFitness":882.4690551757812,"meanFitness":269.5255101835355,"acoFood":93,"randomFood":0,"sigma":0.433156813290537,"species":1},{"generation":15,"bestFood":40.333333333333336,"championFood":40.333333333333336,"bestFitness":847.47216796875,"meanFitness":253.20536019653082,"acoFood":93,"randomFood":0,"sigma":0.5111250396828336,"species":1},{"generation":16,"bestFood":89,"championFood":89,"bestFitness":825.3905639648438,"meanFitness":316.6470828521997,"acoFood":93,"randomFood":0,"sigma":0.47023503650820697,"species":1},{"generation":17,"bestFood":89,"championFood":89,"bestFitness":1249.147216796875,"meanFitness":226.71539221794228,"acoFood":93,"randomFood":0,"sigma":0.4326162335875504,"species":1},{"generation":18,"bestFood":89,"championFood":89,"bestFitness":923.698974609375,"meanFitness":188.8197231895756,"acoFood":93,"randomFood":0,"sigma":0.4326162335875504,"species":1},{"generation":19,"bestFood":89,"championFood":89,"bestFitness":1407.0440673828125,"meanFitness":330.84430499747396,"acoFood":93,"randomFood":0,"sigma":0.4326162335875504,"species":1},{"generation":20,"bestFood":89,"championFood":89,"bestFitness":1069.6832275390625,"meanFitness":267.66137507371604,"acoFood":93,"randomFood":0,"sigma":0.5104871556333095,"species":1},{"generation":21,"bestFood":40.333333333333336,"championFood":89,"bestFitness":1199.8748779296875,"meanFitness":182.28614131733775,"acoFood":93,"randomFood":0,"sigma":0.6023748436473052,"species":1},{"generation":22,"bestFood":89,"championFood":89,"bestFitness":1437.70751953125,"meanFitness":251.91155350767076,"acoFood":93,"randomFood":0,"sigma":0.7108023155038201,"species":1},{"generation":23,"bestFood":89,"championFood":89,"bestFitness":1144.0455322265625,"meanFitness":264.02883081138134,"acoFood":93,"randomFood":0,"sigma":0.8387467322945077,"species":1},{"generation":24,"bestFood":89,"championFood":89,"bestFitness":961.635009765625,"meanFitness":242.5894801132381,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":25,"bestFood":14.666666666666666,"championFood":89,"bestFitness":1100.554443359375,"meanFitness":228.15572584979236,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":26,"bestFood":38.666666666666664,"championFood":89,"bestFitness":682.4028930664062,"meanFitness":121.56002493947744,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":27,"bestFood":38.666666666666664,"championFood":89,"bestFitness":654.1260375976562,"meanFitness":108.59298836579546,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":28,"bestFood":38.666666666666664,"championFood":89,"bestFitness":738.4699096679688,"meanFitness":142.8534558787942,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":29,"bestFood":47,"championFood":89,"bestFitness":1413.020751953125,"meanFitness":318.35554420854896,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":30,"bestFood":47,"championFood":89,"bestFitness":1190.935546875,"meanFitness":153.76077824644744,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":31,"bestFood":47,"championFood":89,"bestFitness":1138.2052001953125,"meanFitness":205.98042181576602,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":32,"bestFood":40.666666666666664,"championFood":89,"bestFitness":704.6019897460938,"meanFitness":120.90051963025326,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":33,"bestFood":62.666666666666664,"championFood":89,"bestFitness":1048.8089599609375,"meanFitness":199.61383455898613,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":34,"bestFood":54.666666666666664,"championFood":89,"bestFitness":1537.81494140625,"meanFitness":294.2279443802254,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":35,"bestFood":88.33333333333333,"championFood":88.33333333333333,"bestFitness":1759.79443359375,"meanFitness":290.88426970265573,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":36,"bestFood":86,"championFood":88.33333333333333,"bestFitness":1524.6292724609375,"meanFitness":450.16483633732423,"acoFood":93,"randomFood":0,"sigma":0.8280000000000001,"species":1},{"generation":37,"bestFood":86,"championFood":88.33333333333333,"bestFitness":1546.81689453125,"meanFitness":497.2752578552754,"acoFood":93,"randomFood":0,"sigma":0.8280000000000001,"species":1},{"generation":38,"bestFood":88.33333333333333,"championFood":88.33333333333333,"bestFitness":2008.323974609375,"meanFitness":443.51634843088686,"acoFood":93,"randomFood":0,"sigma":0.8280000000000001,"species":1},{"generation":39,"bestFood":88.33333333333333,"championFood":88.33333333333333,"bestFitness":1821.492919921875,"meanFitness":567.1001184955239,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":40,"bestFood":88.33333333333333,"championFood":88.33333333333333,"bestFitness":1812.63232421875,"meanFitness":462.65297539159656,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":41,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1532.18212890625,"meanFitness":439.9488580254838,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":42,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1445.5234375,"meanFitness":267.4834160148166,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":43,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1640.684326171875,"meanFitness":469.10982270538807,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":44,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1686.3714599609375,"meanFitness":476.0755062252283,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":45,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1500.1988525390625,"meanFitness":452.92092685215175,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":46,"bestFood":91.66666666666667,"championFood":88.33333333333333,"bestFitness":2199.2177734375,"meanFitness":620.9266271358356,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":47,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1460.2125244140625,"meanFitness":294.46390947699547,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":48,"bestFood":59.666666666666664,"championFood":88.33333333333333,"bestFitness":1956.3492431640625,"meanFitness":571.6856693475129,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":49,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1973.50146484375,"meanFitness":443.5003269314766,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":50,"bestFood":88.33333333333333,"championFood":88.33333333333333,"bestFitness":1710.0899658203125,"meanFitness":476.4256767630577,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":51,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1683.0682373046875,"meanFitness":378.11319479346275,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":52,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1531.4591064453125,"meanFitness":397.0819876836613,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":53,"bestFood":85,"championFood":88.33333333333333,"bestFitness":1542.24853515625,"meanFitness":476.9557934552431,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":54,"bestFood":117,"championFood":117,"bestFitness":2031.641845703125,"meanFitness":588.896680302918,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":55,"bestFood":117,"championFood":117,"bestFitness":2006.91552734375,"meanFitness":627.7809993959963,"acoFood":93,"randomFood":0,"sigma":0.8280000000000001,"species":1},{"generation":56,"bestFood":85,"championFood":117,"bestFitness":1977.7882080078125,"meanFitness":701.3936447426677,"acoFood":93,"randomFood":0,"sigma":0.8280000000000001,"species":1},{"generation":57,"bestFood":85,"championFood":117,"bestFitness":1684.4677734375,"meanFitness":560.2594396099448,"acoFood":93,"randomFood":0,"sigma":0.8280000000000001,"species":1},{"generation":58,"bestFood":85,"championFood":117,"bestFitness":1541.7403564453125,"meanFitness":498.5496258735657,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":59,"bestFood":53,"championFood":117,"bestFitness":2004.721435546875,"meanFitness":521.9726041704416,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":60,"bestFood":85,"championFood":117,"bestFitness":1877.7669677734375,"meanFitness":657.3633392136544,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":61,"bestFood":53,"championFood":117,"bestFitness":1918.5611572265625,"meanFitness":537.5766248330474,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":62,"bestFood":63.666666666666664,"championFood":117,"bestFitness":2308.575439453125,"meanFitness":1026.4909696280956,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":63,"bestFood":63.666666666666664,"championFood":117,"bestFitness":2044.1844482421875,"meanFitness":843.9059919677675,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":64,"bestFood":63.666666666666664,"championFood":117,"bestFitness":1911.071044921875,"meanFitness":701.8804492149502,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":65,"bestFood":58.666666666666664,"championFood":117,"bestFitness":2864.953125,"meanFitness":1188.9292594268918,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":66,"bestFood":131,"championFood":131,"bestFitness":2177.341064453125,"meanFitness":910.4446178749204,"acoFood":93,"randomFood":0,"sigma":0.9,"species":1},{"generation":67,"bestFood":132,"championFood":167.33333333333334,"bestFitness":3086.079833984375,"meanFitness":2213.4721755981445,"acoFood":93,"randomFood":0,"sigma":0.16,"species":1},{"generation":68,"bestFood":125.33333333333333,"championFood":167.33333333333334,"bestFitness":3119.05712890625,"meanFitness":2292.2619094848633,"acoFood":93,"randomFood":0,"sigma":0.16,"species":1},{"generation":69,"bestFood":162.33333333333334,"championFood":167.33333333333334,"bestFitness":3075.97900390625,"meanFitness":1912.830005645752,"acoFood":93,"randomFood":0,"sigma":0.16,"species":1},{"generation":70,"bestFood":132,"championFood":167.33333333333334,"bestFitness":3290.6533203125,"meanFitness":2257.1633071899414,"acoFood":93,"randomFood":0,"sigma":0.1888,"species":1},{"generation":71,"bestFood":118,"championFood":167.33333333333334,"bestFitness":3609.821044921875,"meanFitness":2440.33292388916,"acoFood":93,"randomFood":0,"sigma":0.22278399999999998,"species":1},{"generation":72,"bestFood":161.66666666666666,"championFood":167.33333333333334,"bestFitness":2957.0126953125,"meanFitness":2139.7674560546875,"acoFood":93,"randomFood":0,"sigma":0.26288511999999997,"species":1},{"generation":73,"bestFood":63.666666666666664,"championFood":167.33333333333334,"bestFitness":2768.488037109375,"meanFitness":1985.4692339897156,"acoFood":93,"randomFood":0,"sigma":0.31020444159999994,"species":1},{"generation":74,"bestFood":161.66666666666666,"championFood":167.33333333333334,"bestFitness":2866.08642578125,"meanFitness":1719.3712520599365,"acoFood":93,"randomFood":0,"sigma":0.3660412410879999,"species":1},{"generation":75,"bestFood":76.66666666666667,"championFood":167.33333333333334,"bestFitness":2888.4052734375,"meanFitness":1428.111713886261,"acoFood":93,"randomFood":0,"sigma":0.4319286644838399,"species":1},{"generation":76,"bestFood":161.66666666666666,"championFood":167.33333333333334,"bestFitness":3266.91259765625,"meanFitness":1621.140015937388,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":77,"bestFood":142.33333333333334,"championFood":167.33333333333334,"bestFitness":3150.534912109375,"meanFitness":1810.7922206744552,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":78,"bestFood":161.66666666666666,"championFood":167.33333333333334,"bestFitness":3000.50244140625,"meanFitness":1738.642662981525,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":79,"bestFood":121,"championFood":167.33333333333334,"bestFitness":3304.749267578125,"meanFitness":1931.017512254417,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":80,"bestFood":161.66666666666666,"championFood":167.33333333333334,"bestFitness":3409.939453125,"meanFitness":1994.500333994627,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":81,"bestFood":87.33333333333333,"championFood":167.33333333333334,"bestFitness":3395.162353515625,"meanFitness":1541.4728582426906,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":82,"bestFood":168,"championFood":167.33333333333334,"bestFitness":3174.871337890625,"meanFitness":1886.2466621398926,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":83,"bestFood":163.33333333333334,"championFood":167.33333333333334,"bestFitness":3145.0810546875,"meanFitness":1420.1718747764826,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":84,"bestFood":170.33333333333334,"championFood":167.33333333333334,"bestFitness":3790.976806640625,"meanFitness":2158.877395192161,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":85,"bestFood":184,"championFood":184,"bestFitness":3476.09375,"meanFitness":2173.9432796537876,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":86,"bestFood":184,"championFood":184,"bestFitness":3757.328369140625,"meanFitness":2138.677816092968,"acoFood":93,"randomFood":0,"sigma":0.46,"species":1},{"generation":87,"bestFood":163.33333333333334,"championFood":184,"bestFitness":3586.056884765625,"meanFitness":1641.9130363464355,"acoFood":93,"randomFood":0,"sigma":0.46,"species":1},{"generation":88,"bestFood":209.66666666666666,"championFood":209.66666666666666,"bestFitness":3718.00732421875,"meanFitness":2277.4102080762386,"acoFood":93,"randomFood":0,"sigma":0.46,"species":1},{"generation":89,"bestFood":184,"championFood":209.66666666666666,"bestFitness":3658.663818359375,"meanFitness":2210.6910093277693,"acoFood":93,"randomFood":0,"sigma":0.4232,"species":1},{"generation":90,"bestFood":184,"championFood":209.66666666666666,"bestFitness":3463.36474609375,"meanFitness":2643.602589070797,"acoFood":93,"randomFood":0,"sigma":0.4232,"species":1},{"generation":91,"bestFood":172.66666666666666,"championFood":209.66666666666666,"bestFitness":3672.439697265625,"meanFitness":2293.112373352051,"acoFood":93,"randomFood":0,"sigma":0.4232,"species":1},{"generation":92,"bestFood":135.33333333333334,"championFood":209.66666666666666,"bestFitness":3271.50390625,"meanFitness":2279.129490852356,"acoFood":93,"randomFood":0,"sigma":0.499376,"species":1},{"generation":93,"bestFood":193,"championFood":209.66666666666666,"bestFitness":3529.30126953125,"meanFitness":2461.465042114258,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1},{"generation":94,"bestFood":193,"championFood":209.66666666666666,"bestFitness":4347.47265625,"meanFitness":2465.345762874931,"acoFood":93,"randomFood":0,"sigma":0.5,"species":1}],"benchmark":{"seeds":[9001,9002,9003,9004,9005],"ants":72,"ticks":1200,"rows":[{"id":"trained","label":"Full Trained Brain","kind":"policy","mean":159.6,"std":144.38940404337157,"samples":[4,277,245,0,272],"delta":0},{"id":"aco","label":"Hand-Crafted ACO","kind":"policy","mean":115.2,"std":86.78536743022985,"samples":[23,146,197,22,188],"delta":-44.39999999999999},{"id":"random","label":"Random Brain","kind":"policy","mean":7.2,"std":16.09968943799849,"samples":[0,0,0,0,36],"delta":-152.4},{"id":"foodTrail","label":"-foodTrail","kind":"ablation","mean":136,"std":127.30082482058002,"samples":[5,194,196,0,285],"delta":-23.599999999999994},{"id":"homeGuide","label":"-homeGuide","kind":"ablation","mean":90.8,"std":67.4366369268219,"samples":[38,125,147,0,144],"delta":-68.8},{"id":"alarm","label":"-alarm","kind":"ablation","mean":126,"std":110.21796586763884,"samples":[23,152,193,3,259],"delta":-33.599999999999994},{"id":"scoutPulse","label":"-scoutPulse","kind":"ablation","mean":42.2,"std":38.67428085950662,"samples":[1,81,63,0,66],"delta":-117.39999999999999},{"id":"noEntry","label":"-noEntry","kind":"ablation","mean":147.6,"std":121.13752515220047,"samples":[41,188,282,1,226],"delta":-12},{"id":"pathIntegration","label":"-PathIntegration","kind":"ablation","mean":24,"std":29.146183283579344,"samples":[0,33,17,0,70],"delta":-135.6}]},"weights":[-1.1417,0.7804,0.242,1.5661,1.3834,-1.6719,0.9968,-3.2339,-1.4682,-1.0929,2.0603,0.3253,-0.2858,-2.5858,0.4445,-0.9031,-0.9908,1.2627,-0.3912,-1.7325,-2.0993,0.8119,-0.8852,-0.4567,1.2918,0.6935,1.2691,0.2156,1.2792,1.4495,-1.1586,-0.6687,-0.3771,0.3771,0.7586,1.2398,-1.1685,-0.105,-1.0987,-0.5064,-2.4476,1.8111,-1.5645,-1.123,-1.5535,0.1422,-0.6412,0.2696,-0.4044,-1.3701,1.5181,0.829,0.1794,-1.3082,0.5629,1.0927,0.517,-0.2099,3.2122,0.8568,-0.1631,1.1294,-0.0654,-0.2902,0.6323,0.0877,-0.7536,1.7746,-1.1262,-1.0075,0.468,1.3961,-1.2573,2.939,-1.7337,0.5194,-0.0607,-0.8519,2.8706,0.8192,-3.1913,1.4814,-0.0876,-0.0905,-1.177,2.341,1.7183,1.0526,-0.0807,-0.5278,0.1478,1.4275,0.2456,-0.8946,0.3431,0.0689,0.6696,0.8847,0.0369,-0.7502,-3.385,-0.1859,-0.1115,-2.2719,-2.1797,0.9202,0.1739,-0.6574,-0.7899,-1.3802,-0.9005,-0.343,0.2058,0.5033,-2.8412,-0.4209,-0.1581,0.7726,-0.7219,-0.7325,3.2667,-2.1135,-0.0104,0.7633,-0.9509,-2.3192,0.6646,-2.7069,0.6325,2.3902,-0.4091,-2.1472,-1.0763,-0.8099,-0.3353,2.5751,-1.3989,-1.5593,-0.8516,2.0957,-1.2999,2.9214,-0.4906,-0.8434,0.5358,-0.5123,-0.4092,2.7849,0.199,-0.9372,-0.8474,-1.0481,2.5319,-0.8256,1.2777,-0.5889,1.0783,1.7692,0.6013,-3.08,0.6714,2.228,-1.4446,-0.3642,-0.4583,1.3616,0.5625,2.6378,-0.7072,-1.6896,-0.5711,0.3562,2.531,-0.2014,0.4774,-1.7244,0.4082,-2.1441,0.4622,-1.0582,0.917,-1.6895,0.6052,0.4381,1.7529,-0.1621,0.7121,2.7137,0.0103,-0.4108,0.3273,-0.1002,0.0363,0.3452,-0.2311,-0.4973,-2.2706,-0.7349,3.7098,-1.5862,2.7923,-1.4231,0.0195,-1.3268,1.908,-0.9576,0.7831,-0.3485,0.0134,-1.6673,-0.1323,0.0507,-1.8604,-0.8835,0.2148,-1.3471,0.6312,-0.392,0.1743,0.1008,-1.7373,1.0411,-1.6141,1.0348,-0.858,-0.0306,1.0034,1.1717,0.5477,-0.2693,-0.9137,0.1626,0.321,-2.8398,0.9204,0.7353,-2.1549,0.5524,0.7769,-1.9147,0.3322,-0.3171,0.7131,-1.876,-0.5618,-0.7283,0.7896,0.0749,0.0915,0.3546,0.9268,-0.4156,1.835,-4.0732,-0.4999,-0.8961,0.4894,0.0161,-1.2561,-0.7168,0.4175,0.8273,0.8679,-0.4459,-0.6396,-0.9325,1.2744,3.614,0.8684,-0.2032,2.6022,0.9951,0.261,-0.6642,2.7117,1.6864,-0.3247,-1.5235,-0.2747,-0.9739,0.0219,-1.8208,2.2089,0.7319,1.4242,-1.4296,-1.2774,-0.6941,-0.6417,-0.7084,-0.2133,2.9975,0.7984,-0.5205,1.5047,0.953,-1.213,-0.4505,-1.7537,-1.1909,-1.4502,0.4453,0.6933,-0.079,1.5206,2.3238,0.0878,-0.5934,-0.5319,-0.68,-0.0395,-0.8997,2.7917,0.134,0.3956,-1.5876,1.0309,0.0066,-1.3843,0.2086,-0.0427,0.7294,-0.4098,0.5644,0.0881,0.5363,-0.485,0.9686,0.8474,-0.8659,1.4983,-0.2769,0.124,1.6394,1.2596,-0.9566,0.3495,-0.6237,0.8715,-0.2255,-0.5018,0.1889,-0.7441,-2.1409,3.0257,-1.5249,-1.5388,1.2859,0.3043,-4.296,1.6752,-0.754,-0.5827,0.3255,-0.0359,0.7587,-1.8016,-4.366,1.1737,1.2145,-0.562,-2.9892,-0.083,-1.5434,-0.9442,-0.7514,-1.8742,-1.3598,0.6165,0.847,-1.1442,-1.2427,-0.0421,-0.922,-1.1245,1.0144,0.4391,-0.6905,-1.1856,-0.4095,-0.283,-1.9735,-0.7145,-3.1029,-0.4633,0.0624,-0.5838,-0.9702,-0.8218,0.8626,-0.6241,-0.8764,-2.6226,-0.3488,-0.1764,-1.97,0.9917,1.5695,-0.7447,3.1949,1.4221,-1.5355,-0.8603,-1.2074,0.6453,0.7363,1.4216,-1.902,0.0076,-1.5873,-0.2712,0.9547,-0.5107,-1.1551,0.4776,1.6323,1.6229,-1.7892,1.1348,-0.0314,0.1794,-0.0181,1.9393,0.0648,0.0577,0.2026,-0.2428,1.3529,1.18,3.2159,-0.8355,-0.9816,-0.3641,0.5493,0.4515,1.3922,-0.5238,-1.9365,2.4653,-0.1742,-0.6218,-1.0934,-1.446,-0.5761,0.154,-0.1254,-0.7023,1.0145,0.055,0.1357,-0.1296,-0.3445,0.1313,-2.9996,-0.2146,-2.3448,-0.025,0.1512,0.9179,1.5123,0.0981,-0.2699,0.5065,-1.0024,-0.0262,0.815,0.9843,-0.1109,-0.4685,1.2543,-0.1785,1.2616,0.8948]};
/* @@CHAMPION_END@@ */

export function hasBakedChampion(): boolean {
  return BAKED_CHAMPION.weights.length === GENOME_SIZE;
}

/** Champion weights if baked, otherwise a deterministic untrained genome. Always a fresh copy. */
export function getChampionWeights(): Float32Array {
  if (hasBakedChampion()) return Float32Array.from(BAKED_CHAMPION.weights);
  return createRandomGenome(new Rng(7), DEFAULT_EVO.initScale);
}

/** Pure source transform used by the CLI trainer to bake a champion into this file. */
export function bakeChampionSource(source: string, champion: BakedChampion): string {
  const a = source.lastIndexOf(CHAMPION_BEGIN);
  const b = source.lastIndexOf(CHAMPION_END);
  if (a < 0 || b < 0 || b < a) throw new Error('champion markers not found');
  const body = `${CHAMPION_BEGIN}\nexport const BAKED_CHAMPION: BakedChampion = ${JSON.stringify(champion)};\n`;
  return source.slice(0, a) + body + source.slice(b);
}
