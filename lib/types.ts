/**
 * Shared type surface for the Cryo-Obsidian ant sanctuary.
 * Everything in the hot loop is Structure-of-Arrays; these interfaces describe those arrays.
 */

/* ------------------------------------------------------------------ */
/* Pheromone channels                                                  */
/* ------------------------------------------------------------------ */

export const CHANNEL_COUNT = 5;
export const CH_FOOD = 0;
export const CH_HOME = 1;
export const CH_ALARM = 2;
export const CH_SCOUT = 3;
export const CH_NOENTRY = 4;

export type ChannelIndex = 0 | 1 | 2 | 3 | 4;

export const CHANNEL_KEYS = ['foodTrail', 'homeGuide', 'alarm', 'scoutPulse', 'noEntry'] as const;
export type ChannelKey = (typeof CHANNEL_KEYS)[number];

export const CHANNEL_NAMES: readonly string[] = [
  'Food Amber',
  'Home Cyan',
  'Alarm Crimson',
  'Scout Violet',
  'No-Entry Viridian',
];

/** Solar Amber, Abyssal Cyan, Crimson-Neon, Electric Violet, Acid Viridian. */
export const CHANNEL_RGB: ReadonlyArray<readonly [number, number, number]> = [
  [255, 168, 0],
  [0, 225, 255],
  [255, 12, 75],
  [175, 50, 255],
  [80, 255, 100],
];

export const CHANNEL_CSS: readonly string[] = [
  'rgb(255,168,0)',
  'rgb(0,225,255)',
  'rgb(255,12,75)',
  'rgb(175,50,255)',
  'rgb(80,255,100)',
];

/* ------------------------------------------------------------------ */
/* Ant enums (plain numeric constants: safe with isolatedModules)      */
/* ------------------------------------------------------------------ */

export const STATE_FORAGING = 0;
export const STATE_RETURNING = 1;
export const STATE_ALARMED = 2;
export const STATE_DEFENDING = 3;
export const STATE_TRAIL_DAMPING = 4;
export const STATE_TANDEM_FOLLOW = 5;
export const STATE_NECROPHORESIS = 6;
export const STATE_DEAD = 7;

export const STATE_NAMES: readonly string[] = [
  'FORAGING',
  'RETURNING',
  'ALARMED',
  'DEFENDING',
  'TRAIL_DAMPING',
  'TANDEM_FOLLOW',
  'NECROPHORESIS',
  'DEAD',
];

export const CASTE_SCOUT = 0;
export const CASTE_HAULER = 1;
export const CASTE_WARDEN = 2;

export const POLICY_TRAINED = 0;
export const POLICY_ACO = 1;
export const POLICY_RANDOM = 2;
export type PolicyType = 0 | 1 | 2;

export const POLICY_NAMES: readonly string[] = ['TRAINED_MLP', 'HANDCRAFTED_ACO', 'RANDOM_MLP'];

/* ------------------------------------------------------------------ */
/* Structure-of-Arrays                                                 */
/* ------------------------------------------------------------------ */

export interface AntSoA {
  x: Float32Array;
  y: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  angle: Float32Array;
  /** Path-integrated vector from the ant back to its nest. */
  homeDx: Float32Array;
  homeDy: Float32Array;
  /** Remembered crystal coordinates (tandem running payload). */
  targetX: Float32Array;
  targetY: Float32Array;
  legPhase: Float32Array;
  pheromoneTimer: Float32Array;
  /** Seconds since the nest was last visited (home gland freshness). */
  timeSinceLastVisit: Float32Array;
  ouNoise: Float32Array;
  energy: Float32Array;
  age: Float32Array;
  maxLifespan: Float32Array;
  fitness: Float32Array;
  memA: Float32Array;
  memB: Float32Array;
  stateTimer: Float32Array;
  tripDist: Float32Array;
  tripTime: Float32Array;
  tripStraight: Float32Array;
  anchorX: Float32Array;
  anchorY: Float32Array;
  stallTimer: Float32Array;
  alarmCooldown: Float32Array;
  state: Uint8Array;
  caste: Uint8Array;
  policyType: Uint8Array;
  carryingFood: Uint8Array;
  active: Uint8Array;
  cohort: Uint8Array;
  bump: Uint8Array;
  generation: Uint16Array;
  /** Warden -> corpse index, corpse -> carrier index, -1 none, -2 interred. */
  link: Int16Array;
}

export interface PredatorSoA {
  x: Float32Array;
  y: Float32Array;
  angle: Float32Array;
  hp: Float32Array;
  cooldown: Float32Array;
  phase: Float32Array;
  target: Int16Array;
  active: Uint8Array;
}

export interface FoodSoA {
  x: Float32Array;
  y: Float32Array;
  amount: Float32Array;
  initial: Float32Array;
  radius: Float32Array;
  depletedTimer: Float32Array;
  variant: Uint8Array;
  active: Uint8Array;
}

export const ARC_TROPHALLAXIS = 0;
export const ARC_TANDEM = 1;
export const ARC_COMBAT = 2;

export interface ArcBuffer {
  x1: Float32Array;
  y1: Float32Array;
  x2: Float32Array;
  y2: Float32Array;
  kind: Uint8Array;
  ttl: Uint8Array;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/* ------------------------------------------------------------------ */
/* Simulation parameters                                               */
/* ------------------------------------------------------------------ */

export interface SimParams {
  maxPop: number;
  /** Multiplier on metabolic ATP burn. */
  atpBurn: number;
  /** Mean lifespan in simulated seconds. */
  lifespan: number;
  mortality: boolean;
  births: boolean;
  founderWaves: boolean;
}

export type ArenaMode = 'trained' | 'ab-random' | 'ab-aco' | 'aco';

export type AblationId = ChannelKey | 'pathIntegration';

export type PresetId = 'open' | 'bridge' | 'labyrinth';

export const EVENT_FOUNDER = 1;
export const EVENT_GENERATION = 2;
export const EVENT_PREDATOR_SLAIN = 4;
export const EVENT_FIRST_FOOD = 8;

/* ------------------------------------------------------------------ */
/* Shared colony brain                                                 */
/* ------------------------------------------------------------------ */

export interface BrainJSON {
  format: 'cryo-colony-brain';
  version: 1;
  inputs: number;
  hidden: number;
  outputs: number;
  generation: number;
  weights: number[];
}

export interface GenerationRecord {
  generation: number;
  /** Food delivered by this generation's best genome, mean across the fixed curve seeds. */
  bestFood: number;
  /** Best-so-far champion food on the same seeds. */
  championFood: number;
  bestFitness: number;
  meanFitness: number;
  acoFood: number;
  randomFood: number;
  sigma: number;
  species: number;
}

export interface BenchmarkRow {
  id: string;
  label: string;
  kind: 'policy' | 'ablation';
  mean: number;
  std: number;
  samples: number[];
  /** mean - fullTrained.mean (negative = drop). */
  delta: number;
}

export interface BenchmarkResult {
  seeds: number[];
  ants: number;
  ticks: number;
  rows: BenchmarkRow[];
}

export interface BenchmarkProgress {
  done: number;
  total: number;
  label: string;
}

export interface BakedChampion {
  version: number;
  trainedAt: string;
  generations: number;
  populationSize: number;
  trainSeed: number;
  history: GenerationRecord[];
  benchmark: BenchmarkResult | null;
  weights: number[];
}

export interface EpisodeConfig {
  seed: number;
  ants: number;
  ticks: number;
  worldSize: number;
  policy: PolicyType;
  weights: Float32Array | null;
  ablation: AblationId | null;
}

export interface EpisodeResult {
  food: number;
  pickups: number;
  fitness: number;
  efficiency: number;
  deaths: number;
  wallRate: number;
  loops: number;
}

/* ------------------------------------------------------------------ */
/* UI / telemetry                                                      */
/* ------------------------------------------------------------------ */

export interface Telemetry {
  fps: number;
  alive: number;
  deaths: number;
  food: number;
  generation: number;
  births: number;
  predators: number;
  scoreA: number;
  scoreB: number;
  arena: ArenaMode;
  adaptive: boolean;
}

export type ToolId = 'food-trail' | 'home-trail' | 'alarm' | 'crystal' | 'wall' | 'eraser';

/** Channel index 0..4 or 5 = spawn predator. */
export type SubMode = 0 | 1 | 2 | 3 | 4 | 5;

export interface ToastMessage {
  id: number;
  text: string;
  tone: 'info' | 'good' | 'warn';
}
