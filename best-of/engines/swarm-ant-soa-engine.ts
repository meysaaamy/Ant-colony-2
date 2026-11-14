/*
 * BEST-OF SOURCE FILE
 * Source zip: swarm-rl-simulation-engine.zip (swarm-rl-simulation-engine/lib/ant-soa-engine.ts)
 * Why best: NEAT درون خود موتور (NeatNode/NeatConnection با innovation، compileNetwork به فرمت CSR)، HyperGenes (۸ ضریب رفتاری)، hallOfFame، learning[]، bottleneck/turbo، wall-sliding، targetIndex غذا، world Fixed ۷۲۰×۱۲۸۰
 * Borrow from other zips: prismatic: szybki padded grid؛ fable: SoA typed arrays
 * Notes: مرجع جایگزین برای موتور با NEAT结構ی —妥لمynesد در best-of/engines/
 * Status: verbatim copy — adapt import paths when merging into app/
 */

import { PheromoneGrid, DEFAULT_PHEROMONE_SETTINGS } from "./pheromone-grid";
import { getPreset } from "./presets";
import { clamp, XorShift32 } from "./prng-noise";
import {
  AntState,
  Caste,
  Cohort,
  FIXED_DT,
  MAX_ANTS,
  MAX_PREDATORS,
  Pheromone,
  WORLD_HEIGHT,
  WORLD_WIDTH,
  type AntSoA,
  type ColonySettings,
  type ComparisonMode,
  type Corpse,
  type FoodSource,
  type HyperGenes,
  type LearningPoint,
  type NeatConnection,
  type NeatNode,
  type Predator,
  type SharedNeatBrain,
  type SignalArc,
  type Telemetry,
  type Vec2,
  type WorldPreset,
} from "./types";

const MAX_FOOD = 96;
const MAX_CORPSES = 320;
const MAX_ARCS = 128;
const MAX_NEAT_NODES = 48;
const HASH_CELL_SIZE = 24;
const HASH_COLS = Math.ceil(WORLD_WIDTH / HASH_CELL_SIZE);
const HASH_ROWS = Math.ceil(WORLD_HEIGHT / HASH_CELL_SIZE);
const INPUT_COUNT = 14;
const OUTPUT_COUNT = 4;
const TAU = Math.PI * 2;

export const DEFAULT_COLONY_SETTINGS: ColonySettings = {
  maxPopulation: 1600,
  predatorCount: 0,
  atpBurn: 1,
  lifespan: 210,
  scoutRatio: 0.28,
  wardenRatio: 0.18,
};

export function createAntSoA(): AntSoA {
  return {
    x: new Float32Array(MAX_ANTS), y: new Float32Array(MAX_ANTS),
    vx: new Float32Array(MAX_ANTS), vy: new Float32Array(MAX_ANTS),
    angle: new Float32Array(MAX_ANTS), homeDx: new Float32Array(MAX_ANTS),
    homeDy: new Float32Array(MAX_ANTS), targetX: new Float32Array(MAX_ANTS),
    targetY: new Float32Array(MAX_ANTS), legPhase: new Float32Array(MAX_ANTS),
    pheromoneTimer: new Float32Array(MAX_ANTS), timeSinceLastVisit: new Float32Array(MAX_ANTS),
    ouNoise: new Float32Array(MAX_ANTS), energy: new Float32Array(MAX_ANTS),
    age: new Float32Array(MAX_ANTS), maxLifespan: new Float32Array(MAX_ANTS),
    fitness: new Float32Array(MAX_ANTS), memA: new Float32Array(MAX_ANTS),
    memB: new Float32Array(MAX_ANTS), tripDistance: new Float32Array(MAX_ANTS),
    tripOriginDistance: new Float32Array(MAX_ANTS), stallTime: new Float32Array(MAX_ANTS),
    state: new Uint8Array(MAX_ANTS), caste: new Uint8Array(MAX_ANTS),
    cohort: new Uint8Array(MAX_ANTS), carryingFood: new Uint8Array(MAX_ANTS),
    active: new Uint8Array(MAX_ANTS), generation: new Uint16Array(MAX_ANTS),
    targetIndex: new Int32Array(MAX_ANTS),
  };
}

function defaultHyperGenes(): HyperGenes {
  return {
    trailCommitment: 0.72,
    curiosityDrive: 0.55,
    obstacleForesight: 0.65,
    vectorMemoryTrust: 0.8,
    negativeFeedbackSensitivity: 0.65,
    metabolicEfficiency: 0.55,
    trophallaxisAltruism: 0.6,
    alarmAggression: 0.72,
  };
}

export function createInitialBrain(seed = 8128): SharedNeatBrain {
  const random = new XorShift32(seed);
  const nodes: NeatNode[] = [];
  const connections: NeatConnection[] = [];
  for (let id = 0; id < INPUT_COUNT; id += 1) {
    nodes.push({ id, kind: "input", bias: 0, activation: "tanh" });
  }
  for (let id = INPUT_COUNT; id < INPUT_COUNT + OUTPUT_COUNT; id += 1) {
    nodes.push({ id, kind: "output", bias: random.signed() * 0.12, activation: id === 15 ? "sigmoid" : "tanh" });
  }
  let innovation = 0;
  for (let output = INPUT_COUNT; output < INPUT_COUNT + OUTPUT_COUNT; output += 1) {
    for (let input = 0; input < INPUT_COUNT; input += 1) {
      const structuralBias = output === 14 && input < 10 ? 0.16 : 0;
      connections.push({
        from: input,
        to: output,
        weight: random.signed() * 0.55 + structuralBias,
        enabled: true,
        innovation: innovation,
      });
      innovation += 1;
    }
  }
  return {
    nodes,
    connections,
    generation: 1,
    nextNodeId: INPUT_COUNT + OUTPUT_COUNT,
    nextInnovation: innovation,
    hyperGenes: defaultHyperGenes(),
    fitness: 0,
    species: 0,
  };
}

export function cloneBrain(brain: SharedNeatBrain): SharedNeatBrain {
  return {
    nodes: brain.nodes.map((node) => ({ ...node })),
    connections: brain.connections.map((connection) => ({ ...connection })),
    generation: brain.generation,
    nextNodeId: brain.nextNodeId,
    nextInnovation: brain.nextInnovation,
    hyperGenes: { ...brain.hyperGenes },
    fitness: brain.fitness,
    species: brain.species,
  };
}

function activate(value: number, kind: NeatNode["activation"]): number {
  if (kind === "sigmoid") return 1 / (1 + Math.exp(-value));
  if (kind === "relu") return Math.max(0, value);
  return Math.tanh(value);
}

export function evaluateNeat(
  brain: SharedNeatBrain,
  inputs: Float32Array,
  outputs: Float32Array,
  scratch: Float32Array,
): void {
  const count = Math.min(brain.nextNodeId, scratch.length);
  scratch.fill(0, 0, count);
  const nodes = brain.nodes;
  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes[index];
    if (node.id >= scratch.length) continue;
    if (node.kind === "input") scratch[node.id] = inputs[node.id] ?? 0;
    else scratch[node.id] += node.bias;
  }
  const connections = brain.connections;
  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex += 1) {
    const node = nodes[nodeIndex];
    if (node.id >= scratch.length) continue;
    const value = node.kind === "input" ? scratch[node.id] : activate(scratch[node.id], node.activation);
    scratch[node.id] = value;
    for (let connectionIndex = 0; connectionIndex < connections.length; connectionIndex += 1) {
      const connection = connections[connectionIndex];
      if (!connection.enabled || connection.from !== node.id || connection.to >= scratch.length) continue;
      scratch[connection.to] += value * connection.weight;
    }
  }
  for (let output = 0; output < OUTPUT_COUNT; output += 1) outputs[output] = scratch[INPUT_COUNT + output];
}

function angleDifference(target: number, current: number): number {
  let difference = target - current;
  while (difference > Math.PI) difference -= TAU;
  while (difference < -Math.PI) difference += TAU;
  return difference;
}

function makeFood(): FoodSource {
  return { x: 0, y: 0, amount: 0, maxAmount: 1, radius: 20, variant: 0, discovered: false, active: false };
}
function makeCorpse(): Corpse {
  return { x: 0, y: 0, age: 0, caste: 0, active: false, claimedBy: -1 };
}
function makePredator(): Predator {
  return { x: 0, y: 0, vx: 0, vy: 0, angle: 0, hp: 100, maxHp: 100, attackCooldown: 0, active: false, target: -1 };
}
function makeArc(): SignalArc {
  return { x1: 0, y1: 0, x2: 0, y2: 0, ttl: 0, kind: 0, active: false };
}

interface CompiledNetwork {
  nodes: NeatNode[];
  starts: Int32Array;
  targets: Int32Array;
  weights: Float32Array;
  enabled: Uint8Array;
  nodeCount: number;
}

function compileNetwork(brain: SharedNeatBrain): CompiledNetwork {
  const order = new Int32Array(MAX_NEAT_NODES);
  order.fill(-1);
  for (let index = 0; index < brain.nodes.length; index += 1) order[brain.nodes[index].id] = index;
  const connections = brain.connections.slice().sort((a, b) => order[a.from] - order[b.from]);
  const starts = new Int32Array(brain.nodes.length + 1);
  const targets = new Int32Array(connections.length);
  const weights = new Float32Array(connections.length);
  const enabled = new Uint8Array(connections.length);
  for (let index = 0; index < connections.length; index += 1) {
    const connection = connections[index];
    const source = order[connection.from];
    if (source >= 0) starts[source + 1] += 1;
    targets[index] = connection.to;
    weights[index] = connection.weight;
    enabled[index] = connection.enabled ? 1 : 0;
  }
  for (let index = 0; index < brain.nodes.length; index += 1) starts[index + 1] += starts[index];
  return { nodes: brain.nodes, starts, targets, weights, enabled, nodeCount: brain.nextNodeId };
}

export class AntSoAEngine {
  readonly ants = createAntSoA();
  readonly foods: FoodSource[] = Array.from({ length: MAX_FOOD }, makeFood);
  readonly corpses: Corpse[] = Array.from({ length: MAX_CORPSES }, makeCorpse);
  readonly predators: Predator[] = Array.from({ length: MAX_PREDATORS }, makePredator);
  readonly arcs: SignalArc[] = Array.from({ length: MAX_ARCS }, makeArc);
  readonly grid = new PheromoneGrid(WORLD_WIDTH, WORLD_HEIGHT, 6);
  readonly walls = new Uint8Array(this.grid.length);
  readonly nest: Vec2 = { x: 360, y: 1020 };
  readonly graveyard: Vec2 = { x: 640, y: 1160 };
  readonly colony: ColonySettings = { ...DEFAULT_COLONY_SETTINGS };
  readonly pheromones = { ...DEFAULT_PHEROMONE_SETTINGS };
  readonly learning: LearningPoint[] = [];
  brain = createInitialBrain();
  randomBrain = createInitialBrain(99173);
  preset: WorldPreset;
  generation = 1;
  totalDeaths = 0;
  totalBirths = 0;
  totalFood = 0;
  trainedFood = 0;
  randomFood = 0;
  acoFood = 0;
  foodThisGeneration = 0;
  acoThisGeneration = 0;
  randomThisGeneration = 0;
  ticks = 0;
  alive = 0;
  meanEnergy = 100;
  bottleneck = false;
  comparison: ComparisonMode = "trained";
  wallsRevision = 0;
  wallDirty = { minX: 0, minY: 0, maxX: WORLD_WIDTH, maxY: WORLD_HEIGHT };
  private readonly seed: number;
  private random: XorShift32;
  private readonly hashHead = new Int32Array(HASH_COLS * HASH_ROWS);
  private readonly hashNext = new Int32Array(MAX_ANTS);
  private readonly deliveryWindow = new Uint16Array(600);
  private readonly inputs = new Float32Array(INPUT_COUNT);
  private readonly outputs = new Float32Array(OUTPUT_COUNT);
  private readonly networkScratch = new Float32Array(MAX_NEAT_NODES);
  private trainedNetwork!: CompiledNetwork;
  private randomNetwork!: CompiledNetwork;
  private readonly hallOfFame: SharedNeatBrain[] = [];
  private foodBirthProgress = 0;
  get storedFood(): number { return this.foodBirthProgress; }
  private birthsThisGeneration = 0;
  private arcCursor = 0;
  private corpseCursor = 0;
  private bottleneckTimer = 0;
  private unablatedEfficiency = 0;
  private diffusionStride = 1;
  private turbo = false;

  constructor(preset: WorldPreset = getPreset("sanctuary"), seed = 42042) {
    this.preset = preset;
    this.seed = seed;
    this.random = new XorShift32(seed);
    this.grid.setWalls(this.walls);
    this.reset(preset);
  }

  reset(preset: WorldPreset = this.preset): void {
    this.preset = preset;
    this.random = new XorShift32(this.seed);
    this.nest.x = preset.nest.x;
    this.nest.y = preset.nest.y;
    this.graveyard.x = preset.graveyard.x;
    this.graveyard.y = preset.graveyard.y;
    this.ants.active.fill(0);
    this.walls.fill(0);
    this.grid.clear();
    this.generation = 1;
    this.totalDeaths = 0;
    this.totalBirths = 0;
    this.totalFood = 0;
    this.trainedFood = 0;
    this.randomFood = 0;
    this.acoFood = 0;
    this.foodThisGeneration = 0;
    this.acoThisGeneration = 0;
    this.randomThisGeneration = 0;
    this.ticks = 0;
    this.foodBirthProgress = 0;
    this.birthsThisGeneration = 0;
    this.bottleneck = false;
    this.bottleneckTimer = 0;
    this.unablatedEfficiency = 0;
    this.deliveryWindow.fill(0);
    this.brain = createInitialBrain();
    this.randomBrain = createInitialBrain(99173);
    this.trainedNetwork = compileNetwork(this.brain);
    this.randomNetwork = compileNetwork(this.randomBrain);
    this.hallOfFame.length = 0;
    this.learning.length = 0;
    this.learning.push({ generation: 1, neat: 0, aco: this.acoReference(1), random: 0, hiddenNodes: 0, synapses: this.brain.connections.length });
    for (let index = 0; index < MAX_FOOD; index += 1) this.foods[index].active = false;
    for (let index = 0; index < MAX_CORPSES; index += 1) this.corpses[index].active = false;
    for (let index = 0; index < MAX_ARCS; index += 1) this.arcs[index].active = false;
    for (let index = 0; index < MAX_PREDATORS; index += 1) this.predators[index].active = false;
    this.rasterizePresetWalls();
    for (let index = 0; index < preset.food.length; index += 1) {
      const source = preset.food[index];
      this.addFoodSource(source.x, source.y, source.amount, source.radius);
    }
    for (let index = 0; index < preset.initialPopulation; index += 1) this.spawnAnt(false);
    this.applyCohorts();
    for (let index = 0; index < this.colony.predatorCount; index += 1) this.spawnPredator();
    this.wallsRevision += 1;
    this.wallDirty = { minX: 0, minY: 0, maxX: WORLD_WIDTH, maxY: WORLD_HEIGHT };
    this.recountLiving();
  }

  private rasterizePresetWalls(): void {
    for (const wall of this.preset.walls) this.rasterizeWall(wall.x, wall.y, wall.x2, wall.y2, wall.radius, false);
  }

  private rasterizeWall(x: number, y: number, x2: number, y2: number, radius: number, erase: boolean): void {
    const minX = Math.max(0, Math.floor((Math.min(x, x2) - radius) / this.grid.cellSize));
    const maxX = Math.min(this.grid.cols - 1, Math.ceil((Math.max(x, x2) + radius) / this.grid.cellSize));
    const minY = Math.max(0, Math.floor((Math.min(y, y2) - radius) / this.grid.cellSize));
    const maxY = Math.min(this.grid.rows - 1, Math.ceil((Math.max(y, y2) + radius) / this.grid.cellSize));
    const segmentX = x2 - x;
    const segmentY = y2 - y;
    const segmentLengthSquared = Math.max(0.001, segmentX * segmentX + segmentY * segmentY);
    for (let row = minY; row <= maxY; row += 1) {
      for (let column = minX; column <= maxX; column += 1) {
        const px = (column + 0.5) * this.grid.cellSize;
        const py = (row + 0.5) * this.grid.cellSize;
        const t = clamp(((px - x) * segmentX + (py - y) * segmentY) / segmentLengthSquared, 0, 1);
        const dx = px - (x + segmentX * t);
        const dy = py - (y + segmentY * t);
        if (dx * dx + dy * dy <= radius * radius) this.walls[this.grid.index(column, row)] = erase ? 0 : 1;
      }
    }
  }

  paintWall(x: number, y: number, radius: number, erase: boolean): void {
    this.rasterizeWall(x, y, x, y, radius, erase);
    const margin = radius + 18;
    if (this.wallDirty.minX > this.wallDirty.maxX) {
      this.wallDirty.minX = x - margin;
      this.wallDirty.minY = y - margin;
      this.wallDirty.maxX = x + margin;
      this.wallDirty.maxY = y + margin;
    } else {
      this.wallDirty.minX = Math.min(this.wallDirty.minX, x - margin);
      this.wallDirty.minY = Math.min(this.wallDirty.minY, y - margin);
      this.wallDirty.maxX = Math.max(this.wallDirty.maxX, x + margin);
      this.wallDirty.maxY = Math.max(this.wallDirty.maxY, y + margin);
    }
    if (erase) this.grid.erase(x, y, radius);
    this.wallsRevision += 1;
  }

  markWallsClean(): void {
    this.wallDirty.minX = 1;
    this.wallDirty.minY = 0;
    this.wallDirty.maxX = 0;
    this.wallDirty.maxY = 0;
  }

  isWall(x: number, y: number): boolean {
    if (x < 4 || y < 4 || x >= WORLD_WIDTH - 4 || y >= WORLD_HEIGHT - 4) return true;
    const column = clamp(Math.floor(x / this.grid.cellSize), 0, this.grid.cols - 1);
    const row = clamp(Math.floor(y / this.grid.cellSize), 0, this.grid.rows - 1);
    return this.walls[this.grid.index(column, row)] === 1;
  }

  addFoodSource(x: number, y: number, amount = 120, radius = 24): boolean {
    if (this.isWall(x, y)) return false;
    for (let index = 0; index < MAX_FOOD; index += 1) {
      const food = this.foods[index];
      if (food.active) continue;
      food.x = clamp(x, 20, WORLD_WIDTH - 20);
      food.y = clamp(y, 20, WORLD_HEIGHT - 20);
      food.amount = amount;
      food.maxAmount = amount;
      food.radius = radius;
      food.variant = this.random.int(8);
      food.discovered = false;
      food.active = true;
      return true;
    }
    return false;
  }

  clearTrails(): void {
    this.grid.clear();
  }

  setComparison(mode: ComparisonMode): void {
    this.comparison = mode;
    this.trainedFood = 0;
    this.randomFood = 0;
    this.acoFood = 0;
    this.foodThisGeneration = 0;
    this.randomThisGeneration = 0;
    this.acoThisGeneration = 0;
    this.applyCohorts();
  }

  private applyCohorts(): void {
    for (let index = 0; index < MAX_ANTS; index += 1) {
      if (!this.ants.active[index]) continue;
      this.ants.cohort[index] = this.comparison === "trained"
        ? Cohort.TRAINED_NEAT
        : index % 2 === 0 ? Cohort.TRAINED_NEAT : this.comparison === "random" ? Cohort.RANDOM_NET : Cohort.BASELINE_ACO;
    }
  }

  setPredatorCount(count: number): void {
    const target = clamp(Math.round(count), 0, MAX_PREDATORS);
    let active = 0;
    for (const predator of this.predators) if (predator.active) active += 1;
    for (let index = MAX_PREDATORS - 1; index >= 0 && active > target; index -= 1) {
      if (this.predators[index].active) {
        this.predators[index].active = false;
        active -= 1;
      }
    }
    while (active < target) {
      this.spawnPredator();
      active += 1;
    }
    this.colony.predatorCount = target;
  }

  spawnPredator(x?: number, y?: number): boolean {
    const baseX = x ?? WORLD_WIDTH * 0.5;
    const baseY = y ?? WORLD_HEIGHT * 0.36;
    const jitter = x === undefined ? 40 : 0;
    for (let index = 0; index < MAX_PREDATORS; index += 1) {
      const predator = this.predators[index];
      if (predator.active) continue;
      predator.x = clamp(baseX + this.random.signed() * jitter, 40, WORLD_WIDTH - 40);
      predator.y = clamp(baseY + this.random.signed() * jitter, 40, WORLD_HEIGHT - 40);
      predator.vx = 0;
      predator.vy = 0;
      predator.hp = 100;
      predator.maxHp = 100;
      predator.active = true;
      predator.target = -1;
      predator.attackCooldown = 0;
      this.grid.deposit(predator.x, predator.y, Pheromone.alarm, 80, 34);
      let count = 0;
      for (const candidate of this.predators) if (candidate.active) count += 1;
      this.colony.predatorCount = count;
      return true;
    }
    return false;
  }

  private spawnAnt(countBirth = true): boolean {
    const ants = this.ants;
    for (let index = 0; index < Math.min(MAX_ANTS, this.colony.maxPopulation); index += 1) {
      if (ants.active[index]) continue;
      let x = this.nest.x;
      let y = this.nest.y;
      for (let attempt = 0; attempt < 8; attempt += 1) {
        const angle = this.random.next() * TAU;
        const radius = 12 + this.random.next() * 42;
        x = this.nest.x + Math.cos(angle) * radius;
        y = this.nest.y + Math.sin(angle) * radius;
        if (!this.isWall(x, y)) break;
      }
      const casteRoll = this.random.next();
      ants.x[index] = x;
      ants.y[index] = y;
      ants.vx[index] = 0;
      ants.vy[index] = 0;
      ants.angle[index] = this.random.next() * TAU;
      ants.homeDx[index] = this.nest.x - x;
      ants.homeDy[index] = this.nest.y - y;
      ants.targetX[index] = x;
      ants.targetY[index] = y;
      ants.legPhase[index] = this.random.next() * 8;
      ants.pheromoneTimer[index] = this.random.next();
      ants.timeSinceLastVisit[index] = 0;
      ants.ouNoise[index] = this.random.signed();
      ants.energy[index] = 78 + this.random.next() * 22;
      ants.age[index] = 0;
      ants.maxLifespan[index] = this.colony.lifespan * (0.78 + this.random.next() * 0.44);
      ants.fitness[index] = 0;
      ants.memA[index] = 0;
      ants.memB[index] = 0;
      ants.tripDistance[index] = 0;
      ants.tripOriginDistance[index] = 1;
      ants.stallTime[index] = 0;
      ants.state[index] = AntState.FORAGING;
      ants.caste[index] = casteRoll < this.colony.scoutRatio ? Caste.SCOUT : casteRoll > 1 - this.colony.wardenRatio ? Caste.WARDEN : Caste.HAULER;
      ants.cohort[index] = Cohort.TRAINED_NEAT;
      ants.carryingFood[index] = 0;
      ants.active[index] = 1;
      ants.generation[index] = this.generation;
      ants.targetIndex[index] = -1;
      if (countBirth) {
        this.totalBirths += 1;
        this.birthsThisGeneration += 1;
        if (this.birthsThisGeneration >= 25) this.advanceGeneration();
      }
      return true;
    }
    return false;
  }

  private killAnt(index: number, predatorKill = false): void {
    const ants = this.ants;
    if (!ants.active[index] || ants.state[index] === AntState.DEAD) return;
    if (ants.carryingFood[index] === 1) this.dropFood(ants.x[index], ants.y[index]);
    ants.carryingFood[index] = 0;
    ants.state[index] = AntState.DEAD;
    ants.active[index] = 0;
    ants.vx[index] = 0;
    ants.vy[index] = 0;
    this.totalDeaths += 1;
    this.spawnCorpse(ants.x[index], ants.y[index], ants.caste[index]);
    this.grid.deposit(ants.x[index], ants.y[index], Pheromone.alarm, predatorKill ? 80 : 24, predatorKill ? 28 : 15);
  }

  private dropFood(x: number, y: number): void {
    for (let index = 0; index < MAX_FOOD; index += 1) {
      const food = this.foods[index];
      if (food.active && Math.hypot(food.x - x, food.y - y) < 28) {
        food.amount = Math.min(food.maxAmount, food.amount + 1);
        return;
      }
    }
    this.addFoodSource(x, y, 1, 8);
  }

  private spawnCorpse(x: number, y: number, caste: number): void {
    let selected = -1;
    for (let offset = 0; offset < MAX_CORPSES; offset += 1) {
      const index = (this.corpseCursor + offset) % MAX_CORPSES;
      if (!this.corpses[index].active) {
        selected = index;
        break;
      }
    }
    if (selected < 0) selected = this.corpseCursor;
    this.corpseCursor = (selected + 1) % MAX_CORPSES;
    const corpse = this.corpses[selected];
    corpse.x = x;
    corpse.y = y;
    corpse.age = 0;
    corpse.caste = caste;
    corpse.claimedBy = -1;
    corpse.active = true;
  }

  private addArc(x1: number, y1: number, x2: number, y2: number, kind: 0 | 1 | 2): void {
    const arc = this.arcs[this.arcCursor];
    this.arcCursor = (this.arcCursor + 1) % MAX_ARCS;
    arc.x1 = x1;
    arc.y1 = y1;
    arc.x2 = x2;
    arc.y2 = y2;
    arc.kind = kind;
    arc.ttl = 0.12;
    arc.active = true;
  }

  private rebuildSpatialHash(): void {
    this.hashHead.fill(-1);
    for (let index = 0; index < MAX_ANTS; index += 1) {
      if (!this.ants.active[index]) continue;
      const column = clamp(Math.floor(this.ants.x[index] / HASH_CELL_SIZE), 0, HASH_COLS - 1);
      const row = clamp(Math.floor(this.ants.y[index] / HASH_CELL_SIZE), 0, HASH_ROWS - 1);
      const cell = row * HASH_COLS + column;
      this.hashNext[index] = this.hashHead[cell];
      this.hashHead[cell] = index;
    }
  }

  private evaluateCompiled(network: CompiledNetwork): void {
    const scratch = this.networkScratch;
    const count = Math.min(network.nodeCount, scratch.length);
    scratch.fill(0, 0, count);
    const nodes = network.nodes;
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      if (node.id >= scratch.length) continue;
      if (node.kind === "input") scratch[node.id] = this.inputs[node.id];
      else scratch[node.id] += node.bias;
    }
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      if (node.id >= scratch.length) continue;
      const value = node.kind === "input" ? scratch[node.id] : activate(scratch[node.id], node.activation);
      scratch[node.id] = value;
      const end = network.starts[index + 1];
      for (let connection = network.starts[index]; connection < end; connection += 1) {
        if (!network.enabled[connection]) continue;
        const target = network.targets[connection];
        if (target < scratch.length) scratch[target] += value * network.weights[connection];
      }
    }
    for (let output = 0; output < OUTPUT_COUNT; output += 1) this.outputs[output] = scratch[INPUT_COUNT + output];
  }

  private sense(index: number): void {
    const ants = this.ants;
    const angle = ants.angle[index];
    const forwardX = Math.cos(angle);
    const forwardY = Math.sin(angle);
    const sideX = -forwardY;
    const sideY = forwardX;
    const centerX = ants.x[index] + forwardX * 20;
    const centerY = ants.y[index] + forwardY * 20;
    const leftX = centerX + sideX * 12;
    const leftY = centerY + sideY * 12;
    const rightX = centerX - sideX * 12;
    const rightY = centerY - sideY * 12;
    for (let channel = 0; channel < 5; channel += 1) {
      const left = Math.log1p(this.grid.sample(leftX, leftY, channel as 0 | 1 | 2 | 3 | 4));
      const right = Math.log1p(this.grid.sample(rightX, rightY, channel as 0 | 1 | 2 | 3 | 4));
      this.inputs[channel * 2] = left * 0.25;
      this.inputs[channel * 2 + 1] = right * 0.25;
    }
    const frontWall = this.isWall(ants.x[index] + forwardX * 30, ants.y[index] + forwardY * 30) ? 1 : 0;
    const leftWall = this.isWall(ants.x[index] + forwardX * 19 + sideX * 20, ants.y[index] + forwardY * 19 + sideY * 20) ? 1 : 0;
    const rightWall = this.isWall(ants.x[index] + forwardX * 19 - sideX * 20, ants.y[index] + forwardY * 19 - sideY * 20) ? 1 : 0;
    this.inputs[10] = frontWall + (leftWall - rightWall) * 0.5;
    const homeAngle = Math.atan2(ants.homeDy[index], ants.homeDx[index]);
    this.inputs[11] = this.grid.ablation.pathIntegration ? angleDifference(homeAngle, angle) / Math.PI : 0;
    this.inputs[12] = ants.energy[index] / 100 * (ants.carryingFood[index] === 1 ? -1 : 1);
    this.inputs[13] = (ants.memA[index] - ants.memB[index]) * 0.5;
  }

  private updateSocial(index: number): void {
    if (this.ticks % 18 !== index % 18) return;
    const ants = this.ants;
    const column = clamp(Math.floor(ants.x[index] / HASH_CELL_SIZE), 0, HASH_COLS - 1);
    const row = clamp(Math.floor(ants.y[index] / HASH_CELL_SIZE), 0, HASH_ROWS - 1);
    let peer = this.hashHead[row * HASH_COLS + column];
    let inspected = 0;
    while (peer >= 0 && inspected < 10) {
      inspected += 1;
      if (peer !== index && ants.active[peer]) {
        const dx = ants.x[peer] - ants.x[index];
        const dy = ants.y[peer] - ants.y[index];
        const distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < 24 * 24) {
          if (ants.energy[index] > 72 && ants.energy[peer] < 30) {
            const gift = Math.min(10, ants.energy[index] - 55) * this.brain.hyperGenes.trophallaxisAltruism;
            if (gift > 0) {
              ants.energy[index] -= gift;
              ants.energy[peer] = Math.min(100, ants.energy[peer] + gift);
              ants.fitness[index] += gift * 0.15;
              this.addArc(ants.x[index], ants.y[index], ants.x[peer], ants.y[peer], 0);
            }
          }
          if (ants.state[index] === AntState.RETURNING && ants.caste[index] === Caste.SCOUT && ants.state[peer] === AntState.FORAGING) {
            ants.targetX[peer] = ants.targetX[index];
            ants.targetY[peer] = ants.targetY[index];
            ants.targetIndex[peer] = ants.targetIndex[index];
            ants.state[peer] = AntState.TANDEM_FOLLOW;
            ants.memA[peer] = 1;
            this.grid.deposit(ants.x[peer], ants.y[peer], Pheromone.scoutPulse, 16, 12);
            this.addArc(ants.x[index], ants.y[index], ants.x[peer], ants.y[peer], 1);
          }
        }
      }
      peer = this.hashNext[peer];
    }
  }

  private nearestPredator(index: number): number {
    let nearest = -1;
    let best = 150 * 150;
    for (let predatorIndex = 0; predatorIndex < MAX_PREDATORS; predatorIndex += 1) {
      const predator = this.predators[predatorIndex];
      if (!predator.active) continue;
      const dx = predator.x - this.ants.x[index];
      const dy = predator.y - this.ants.y[index];
      const distance = dx * dx + dy * dy;
      if (distance < best) {
        best = distance;
        nearest = predatorIndex;
      }
    }
    return nearest;
  }

  private updateAnt(index: number, dt: number): void {
    const ants = this.ants;
    ants.age[index] += dt;
    ants.timeSinceLastVisit[index] += dt;
    ants.legPhase[index] = (ants.legPhase[index] + Math.hypot(ants.vx[index], ants.vy[index]) * dt * 0.19) % 8;
    if (ants.age[index] >= ants.maxLifespan[index] || ants.energy[index] <= 0) {
      this.killAnt(index);
      return;
    }
    const predatorIndex = this.nearestPredator(index);
    if (predatorIndex >= 0) {
      const predator = this.predators[predatorIndex];
      const dx = predator.x - ants.x[index];
      const dy = predator.y - ants.y[index];
      const distance = Math.hypot(dx, dy);
      if (distance < 75) {
        if ((this.ticks + index) % 18 === 0) this.grid.deposit(ants.x[index], ants.y[index], Pheromone.alarm, 80, 24);
        if (ants.caste[index] === Caste.WARDEN) ants.state[index] = AntState.DEFENDING;
        else if (ants.carryingFood[index] !== 2) ants.state[index] = AntState.ALARMED;
        ants.targetIndex[index] = predatorIndex;
      }
      if (ants.state[index] === AntState.DEFENDING && distance < 25) {
        predator.hp -= (12 + 14 * this.brain.hyperGenes.alarmAggression) * dt;
        ants.energy[index] -= 4.5 * dt;
        ants.fitness[index] += dt * 2;
        if ((this.ticks + index) % 6 === 0) this.addArc(ants.x[index], ants.y[index], predator.x, predator.y, 2);
        if (predator.hp <= 0) {
          predator.active = false;
          this.grid.deposit(predator.x, predator.y, Pheromone.noEntry, 70, 35);
          ants.fitness[index] += 35;
          ants.state[index] = AntState.FORAGING;
          this.colony.predatorCount = Math.max(0, this.colony.predatorCount - 1);
        }
      }
    } else if (ants.state[index] === AntState.ALARMED || ants.state[index] === AntState.DEFENDING) {
      ants.state[index] = ants.carryingFood[index] === 1 ? AntState.RETURNING : AntState.FORAGING;
      ants.targetIndex[index] = -1;
    }

    this.sense(index);
    this.evaluateCompiled(ants.cohort[index] === Cohort.RANDOM_NET ? this.randomNetwork : this.trainedNetwork);
    ants.memA[index] = ants.memA[index] * 0.92 + this.outputs[3] * 0.08;
    ants.memB[index] = ants.memB[index] * 0.96 + this.outputs[0] * 0.04;
    let steering = 0;
    let targetSpeed = ants.caste[index] === Caste.SCOUT ? 56 : ants.caste[index] === Caste.WARDEN ? 48 : 45;
    const hyper = this.brain.hyperGenes;

    if (ants.cohort[index] === Cohort.BASELINE_ACO) {
      steering = clamp((this.inputs[0] - this.inputs[1]) * 1.7 + (this.inputs[2] - this.inputs[3]) * -0.35, -1, 1);
      steering += this.random.signed() * 0.075;
      targetSpeed *= 0.92;
    } else {
      steering = clamp(this.outputs[0] * (0.75 + hyper.curiosityDrive * 0.4), -1.4, 1.4);
      targetSpeed *= 0.42 + this.outputs[1] * 0.78;
      if (ants.cohort[index] === Cohort.TRAINED_NEAT) {
        steering += clamp((this.inputs[0] - this.inputs[1]) * hyper.trailCommitment, -0.65, 0.65);
        steering += clamp((this.inputs[8] - this.inputs[9]) * -hyper.negativeFeedbackSensitivity, -0.45, 0.45);
      }
    }

    if (ants.state[index] === AntState.ALARMED) {
      const predator = this.predators[ants.targetIndex[index]];
      if (predator?.active) {
        const escapeAngle = Math.atan2(ants.y[index] - predator.y, ants.x[index] - predator.x);
        steering = angleDifference(escapeAngle, ants.angle[index]) * 1.5 - (this.inputs[4] - this.inputs[5]) * 0.6 + (this.inputs[2] - this.inputs[3]) * 0.25;
        targetSpeed = 82;
      }
    } else if (ants.state[index] === AntState.DEFENDING) {
      const predator = this.predators[ants.targetIndex[index]];
      if (predator?.active) {
        steering = angleDifference(Math.atan2(predator.y - ants.y[index], predator.x - ants.x[index]), ants.angle[index]) * 1.5;
        steering += (this.inputs[4] - this.inputs[5]) * 0.35;
        targetSpeed = 68;
      }
    } else if (ants.state[index] === AntState.NECROPHORESIS) {
      const carryingCorpse = ants.carryingFood[index] === 2;
      const targetX = carryingCorpse ? this.graveyard.x : ants.targetX[index];
      const targetY = carryingCorpse ? this.graveyard.y : ants.targetY[index];
      steering = angleDifference(Math.atan2(targetY - ants.y[index], targetX - ants.x[index]), ants.angle[index]) * 1.4;
      targetSpeed = 38;
    } else if (ants.state[index] === AntState.TANDEM_FOLLOW) {
      steering += angleDifference(Math.atan2(ants.targetY[index] - ants.y[index], ants.targetX[index] - ants.x[index]), ants.angle[index]) * 1.1;
      targetSpeed = 62;
      if (Math.hypot(ants.targetX[index] - ants.x[index], ants.targetY[index] - ants.y[index]) < 24) ants.state[index] = AntState.FORAGING;
    } else if (ants.state[index] === AntState.TRAIL_DAMPING) {
      targetSpeed = 55;
      this.grid.deposit(ants.x[index], ants.y[index], Pheromone.noEntry, 24 * dt, 9);
      if (ants.stallTime[index] <= 0) ants.state[index] = AntState.FORAGING;
    }

    if (ants.state[index] === AntState.RETURNING || (ants.state[index] === AntState.ALARMED && ants.carryingFood[index] === 1)) {
      const trust = ants.cohort[index] === Cohort.RANDOM_NET ? 0.25 : ants.cohort[index] === Cohort.BASELINE_ACO ? 0.85 : hyper.vectorMemoryTrust;
      if (this.grid.ablation.pathIntegration) {
        const homeBearing = Math.atan2(ants.homeDy[index], ants.homeDx[index]);
        steering += angleDifference(homeBearing, ants.angle[index]) * trust;
      } else {
        steering += (this.inputs[2] - this.inputs[3]) * 0.9 + ants.ouNoise[index] * 0.3;
      }
      targetSpeed *= 0.94;
      this.grid.deposit(ants.x[index], ants.y[index], Pheromone.foodTrail, (10 + this.outputs[2] * 4) * dt, 8);
    } else if (ants.state[index] !== AntState.NECROPHORESIS) {
      this.grid.deposit(ants.x[index], ants.y[index], Pheromone.homeGuide, (3.2 + Math.max(0, this.outputs[2]) * 2) * dt, 6);
    }

    if (ants.state[index] === AntState.FORAGING || ants.state[index] === AntState.TANDEM_FOLLOW) {
      if ((this.ticks + index) % 15 === 0) this.findFood(index);
      if (ants.targetIndex[index] >= 0) {
        const food = this.foods[ants.targetIndex[index]];
        if (food?.active && food.amount > 0) {
          ants.targetX[index] = food.x;
          ants.targetY[index] = food.y;
          const targetBearing = Math.atan2(food.y - ants.y[index], food.x - ants.x[index]);
          steering += angleDifference(targetBearing, ants.angle[index]) * (ants.caste[index] === Caste.SCOUT ? 0.85 : 0.55);
        } else if (food) {
          ants.targetIndex[index] = -1;
          ants.state[index] = AntState.TRAIL_DAMPING;
          ants.stallTime[index] = 0.8;
          ants.angle[index] += Math.PI;
          this.grid.deposit(food.x, food.y, Pheromone.noEntry, 55, 24);
        }
      }
      this.tryHarvest(index);
    }

    if (ants.caste[index] === Caste.WARDEN && ants.state[index] === AntState.FORAGING && (this.ticks + index) % 45 === 0) this.findCorpse(index);
    if (ants.state[index] === AntState.NECROPHORESIS) this.updateCorpseHauling(index);
    this.updateSocial(index);

    if (this.inputs[10] > 0.5) {
      steering += (index % 2 === 0 ? 1 : -1) * (0.8 + hyper.obstacleForesight);
      ants.fitness[index] -= dt * 0.5;
    }
    ants.ouNoise[index] = ants.ouNoise[index] * 0.94 + this.random.signed() * 0.18;
    steering += ants.ouNoise[index] * (0.14 + hyper.curiosityDrive * 0.12);
    ants.angle[index] += clamp(steering, -2.4, 2.4) * dt * 3.1;
    const speed = Math.max(8, targetSpeed);
    const oldX = ants.x[index];
    const oldY = ants.y[index];
    const nextX = oldX + Math.cos(ants.angle[index]) * speed * dt;
    const nextY = oldY + Math.sin(ants.angle[index]) * speed * dt;
    if (!this.isWall(nextX, nextY)) {
      ants.x[index] = nextX;
      ants.y[index] = nextY;
      ants.stallTime[index] = Math.max(0, ants.stallTime[index] - dt * 2);
    } else if (!this.isWall(nextX, oldY)) {
      ants.x[index] = nextX;
      ants.angle[index] += 0.7;
      ants.stallTime[index] += dt;
    } else if (!this.isWall(oldX, nextY)) {
      ants.y[index] = nextY;
      ants.angle[index] -= 0.7;
      ants.stallTime[index] += dt;
    } else {
      ants.angle[index] += Math.PI * (0.55 + this.random.next() * 0.35);
      ants.stallTime[index] += dt * 2;
      ants.fitness[index] -= dt;
    }
    if (ants.stallTime[index] > 1.4 && ants.state[index] === AntState.FORAGING) {
      ants.state[index] = AntState.TRAIL_DAMPING;
      ants.angle[index] += Math.PI;
      ants.stallTime[index] = 0.75;
      this.grid.deposit(oldX, oldY, Pheromone.noEntry, 35, 14);
    }
    ants.vx[index] = (ants.x[index] - oldX) / dt;
    ants.vy[index] = (ants.y[index] - oldY) / dt;
    const moved = Math.hypot(ants.x[index] - oldX, ants.y[index] - oldY);
    ants.homeDx[index] -= ants.x[index] - oldX;
    ants.homeDy[index] -= ants.y[index] - oldY;
    ants.tripDistance[index] += moved;
    const load = ants.carryingFood[index] === 0 ? 0 : 0.055;
    const combat = ants.state[index] === AntState.DEFENDING ? 0.11 : 0;
    const metabolic = 1.15 - hyper.metabolicEfficiency * 0.3;
    ants.energy[index] -= (0.105 + speed * 0.00115 + load + combat) * this.colony.atpBurn * metabolic * dt;

    const homeDistance = Math.hypot(this.nest.x - ants.x[index], this.nest.y - ants.y[index]);
    if (homeDistance < 46) {
      ants.energy[index] = 100;
      ants.homeDx[index] = 0;
      ants.homeDy[index] = 0;
      if (ants.carryingFood[index] === 1) this.deliverFood(index);
      else if (ants.state[index] === AntState.ALARMED) ants.state[index] = AntState.FORAGING;
    }
  }

  private findFood(index: number): void {
    const ants = this.ants;
    if (ants.targetIndex[index] >= 0 && this.foods[ants.targetIndex[index]]?.amount > 0) return;
    let nearest = -1;
    let best = ants.caste[index] === Caste.SCOUT ? 175 * 175 : 115 * 115;
    for (let foodIndex = 0; foodIndex < MAX_FOOD; foodIndex += 1) {
      const food = this.foods[foodIndex];
      if (!food.active || food.amount <= 0) continue;
      const dx = food.x - ants.x[index];
      const dy = food.y - ants.y[index];
      const distance = dx * dx + dy * dy;
      if (distance < best) {
        best = distance;
        nearest = foodIndex;
      }
    }
    if (nearest >= 0) {
      ants.targetIndex[index] = nearest;
      ants.targetX[index] = this.foods[nearest].x;
      ants.targetY[index] = this.foods[nearest].y;
    }
  }

  private tryHarvest(index: number): void {
    const ants = this.ants;
    const foodIndex = ants.targetIndex[index];
    if (foodIndex < 0) return;
    const food = this.foods[foodIndex];
    if (!food.active || food.amount <= 0) return;
    if (Math.hypot(food.x - ants.x[index], food.y - ants.y[index]) > food.radius + 8) return;
    food.amount -= 1;
    ants.carryingFood[index] = 1;
    ants.state[index] = AntState.RETURNING;
    ants.targetX[index] = food.x;
    ants.targetY[index] = food.y;
    ants.tripOriginDistance[index] = Math.max(1, Math.hypot(food.x - this.nest.x, food.y - this.nest.y));
    ants.tripDistance[index] = 1;
    ants.timeSinceLastVisit[index] = 0;
    if (!food.discovered) {
      food.discovered = true;
      ants.fitness[index] += 18;
      this.grid.deposit(food.x, food.y, Pheromone.scoutPulse, 90, 32);
    }
    if (food.amount <= 0) {
      this.grid.deposit(food.x, food.y, Pheromone.noEntry, 80, 30);
      food.active = false;
    }
  }

  private deliverFood(index: number): void {
    const ants = this.ants;
    ants.carryingFood[index] = 0;
    ants.state[index] = AntState.FORAGING;
    const jackpot = 12 * clamp(ants.tripOriginDistance[index] / Math.max(1, ants.tripDistance[index]), 0.05, 1);
    ants.fitness[index] += jackpot;
    ants.targetIndex[index] = -1;
    ants.tripDistance[index] = 0;
    this.totalFood += 1;
    this.deliveryWindow[this.ticks % this.deliveryWindow.length] += 1;
    this.foodBirthProgress += 1;
    const cohort = ants.cohort[index];
    if (cohort === Cohort.TRAINED_NEAT) {
      this.trainedFood += 1;
      this.foodThisGeneration += 1;
    } else if (cohort === Cohort.RANDOM_NET) {
      this.randomFood += 1;
      this.randomThisGeneration += 1;
    } else {
      this.acoFood += 1;
      this.acoThisGeneration += 1;
    }
    if (this.foodBirthProgress >= 3) {
      this.foodBirthProgress -= 3;
      this.spawnAnt(true);
    }
  }

  private findCorpse(index: number): void {
    const ants = this.ants;
    let best = 190 * 190;
    let selected = -1;
    for (let corpseIndex = 0; corpseIndex < MAX_CORPSES; corpseIndex += 1) {
      const corpse = this.corpses[corpseIndex];
      if (!corpse.active || corpse.claimedBy !== -1) continue;
      const dx = corpse.x - ants.x[index];
      const dy = corpse.y - ants.y[index];
      const distance = dx * dx + dy * dy;
      if (distance < best) {
        best = distance;
        selected = corpseIndex;
      }
    }
    if (selected >= 0) {
      const corpse = this.corpses[selected];
      corpse.claimedBy = index;
      ants.targetIndex[index] = selected;
      ants.targetX[index] = corpse.x;
      ants.targetY[index] = corpse.y;
      ants.state[index] = AntState.NECROPHORESIS;
    }
  }

  private updateCorpseHauling(index: number): void {
    const ants = this.ants;
    const corpseIndex = ants.targetIndex[index];
    const corpse = this.corpses[corpseIndex];
    if (!corpse?.active) {
      ants.state[index] = AntState.FORAGING;
      ants.carryingFood[index] = 0;
      ants.targetIndex[index] = -1;
      return;
    }
    if (ants.carryingFood[index] !== 2) {
      if (Math.hypot(corpse.x - ants.x[index], corpse.y - ants.y[index]) < 15) {
        ants.carryingFood[index] = 2;
        corpse.claimedBy = index;
      }
    } else {
      corpse.x = ants.x[index] - Math.cos(ants.angle[index]) * 10;
      corpse.y = ants.y[index] - Math.sin(ants.angle[index]) * 10;
      if (Math.hypot(this.graveyard.x - ants.x[index], this.graveyard.y - ants.y[index]) < 38) {
        corpse.claimedBy = -2;
        corpse.x = this.graveyard.x + this.random.signed() * 26;
        corpse.y = this.graveyard.y + this.random.signed() * 26;
        ants.carryingFood[index] = 0;
        ants.state[index] = AntState.FORAGING;
        ants.targetIndex[index] = -1;
        ants.fitness[index] += 4;
      }
    }
  }

  private updatePredators(dt: number): void {
    const ants = this.ants;
    for (let predatorIndex = 0; predatorIndex < MAX_PREDATORS; predatorIndex += 1) {
      const predator = this.predators[predatorIndex];
      if (!predator.active) continue;
      predator.attackCooldown = Math.max(0, predator.attackCooldown - dt);
      if (this.ticks % 30 === predatorIndex * 3 || predator.target < 0 || !ants.active[predator.target]) {
        let best = Number.POSITIVE_INFINITY;
        predator.target = -1;
        for (let ant = predatorIndex % 3; ant < MAX_ANTS; ant += 3) {
          if (!ants.active[ant]) continue;
          const dx = ants.x[ant] - predator.x;
          const dy = ants.y[ant] - predator.y;
          const distance = dx * dx + dy * dy;
          if (distance < best) {
            best = distance;
            predator.target = ant;
          }
        }
      }
      const target = predator.target;
      if (target >= 0 && ants.active[target]) {
        const dx = ants.x[target] - predator.x;
        const dy = ants.y[target] - predator.y;
        const distance = Math.max(0.01, Math.hypot(dx, dy));
        predator.angle = Math.atan2(dy, dx);
        const speed = distance > 20 ? 37 : 8;
        predator.vx = dx / distance * speed;
        predator.vy = dy / distance * speed;
        if (distance < 20 && predator.attackCooldown <= 0) {
          ants.energy[target] -= 34;
          predator.attackCooldown = 0.7;
          this.grid.deposit(predator.x, predator.y, Pheromone.alarm, 80, 32);
          this.addArc(predator.x, predator.y, ants.x[target], ants.y[target], 2);
          if (ants.energy[target] <= 0) this.killAnt(target, true);
        }
      } else {
        predator.angle += this.random.signed() * dt;
        predator.vx = Math.cos(predator.angle) * 18;
        predator.vy = Math.sin(predator.angle) * 18;
      }
      const nx = clamp(predator.x + predator.vx * dt, 25, WORLD_WIDTH - 25);
      const ny = clamp(predator.y + predator.vy * dt, 25, WORLD_HEIGHT - 25);
      if (!this.isWall(nx, ny)) {
        predator.x = nx;
        predator.y = ny;
      } else predator.angle += Math.PI * 0.7;
    }
  }

  private updateCorpses(dt: number): void {
    for (let index = 0; index < MAX_CORPSES; index += 1) {
      const corpse = this.corpses[index];
      if (!corpse.active) continue;
      corpse.age += dt;
      if (corpse.age < 32 && (this.ticks + index) % 20 === 0) this.grid.deposit(corpse.x, corpse.y, Pheromone.alarm, 1.8, 8);
      if (corpse.age > 240 && corpse.claimedBy === -1) corpse.active = false;
    }
    for (let index = 0; index < MAX_ARCS; index += 1) {
      const arc = this.arcs[index];
      if (!arc.active) continue;
      arc.ttl -= dt;
      if (arc.ttl <= 0) arc.active = false;
    }
  }

  private recountLiving(): void {
    let alive = 0;
    let energy = 0;
    for (let index = 0; index < MAX_ANTS; index += 1) {
      if (!this.ants.active[index]) continue;
      alive += 1;
      energy += this.ants.energy[index];
    }
    this.alive = alive;
    this.meanEnergy = alive ? energy / alive : 0;
  }

  step(dt = FIXED_DT): void {
    this.ticks += 1;
    this.deliveryWindow[this.ticks % this.deliveryWindow.length] = 0;
    if (this.ticks % this.diffusionStride === 0) this.grid.step(dt * this.diffusionStride, this.pheromones);
    this.rebuildSpatialHash();
    for (let index = 0; index < MAX_ANTS; index += 1) {
      if (this.ants.active[index]) this.updateAnt(index, dt);
    }
    this.updatePredators(dt);
    this.updateCorpses(dt);
    if (this.ticks % 30 === 0) this.recountLiving();
    if (this.alive < 35) {
      this.bottleneckTimer += dt;
      if (this.bottleneckTimer > 1) {
        this.bottleneck = true;
        this.bottleneckTimer = 0;
        this.evolveBrain(2.5);
        for (let count = this.alive; count < 48; count += 1) this.spawnAnt(false);
        this.recountLiving();
      }
    } else {
      this.bottleneckTimer = 0;
      if (this.alive > 70) this.bottleneck = false;
    }
  }

  setAdaptiveQuality(low: boolean): void {
    this.diffusionStride = low ? 2 : 1;
  }

  runTurboBatch(ticks = 100): void {
    this.turbo = true;
    const previousStride = this.diffusionStride;
    this.diffusionStride = Math.max(2, previousStride);
    for (let tick = 0; tick < ticks; tick += 1) this.step(FIXED_DT);
    this.diffusionStride = previousStride;
    this.turbo = false;
    this.recountLiving();
  }

  private acoReference(generation: number): number {
    return Math.round(34 + 30 * (1 - Math.exp(-generation / 7)));
  }

  private advanceGeneration(): void {
    let fitness = 0;
    let count = 0;
    for (let index = 0; index < MAX_ANTS; index += 1) {
      if (!this.ants.active[index] || this.ants.cohort[index] !== Cohort.TRAINED_NEAT) continue;
      fitness += this.ants.fitness[index];
      this.ants.fitness[index] *= 0.35;
      count += 1;
    }
    this.brain.fitness = count ? fitness / count : 0;
    this.learning.push({
      generation: this.generation,
      neat: this.foodThisGeneration,
      aco: this.comparison === "aco" ? this.acoThisGeneration : this.acoReference(this.generation),
      random: this.randomThisGeneration,
      hiddenNodes: this.brain.nodes.length - INPUT_COUNT - OUTPUT_COUNT,
      synapses: this.brain.connections.reduce((total, connection) => total + (connection.enabled ? 1 : 0), 0),
    });
    if (this.learning.length > 80) this.learning.shift();
    this.generation += 1;
    this.birthsThisGeneration = 0;
    this.foodThisGeneration = 0;
    this.acoThisGeneration = 0;
    this.randomThisGeneration = 0;
    this.brain.generation = this.generation;
    this.evolveBrain(this.bottleneck ? 2.5 : 1);
  }

  private evolveBrain(mutationScale: number): void {
    if (this.hallOfFame.length > 0 && this.brain.fitness < this.hallOfFame[0].fitness - 1) {
      const generation = this.brain.generation;
      const innovation = Math.max(this.brain.nextInnovation, this.hallOfFame[0].nextInnovation);
      this.brain = cloneBrain(this.hallOfFame[0]);
      this.brain.generation = generation;
      this.brain.nextInnovation = innovation;
    } else if (this.hallOfFame.length === 0 || this.brain.fitness >= this.hallOfFame[0].fitness) {
      this.hallOfFame.unshift(cloneBrain(this.brain));
      if (this.hallOfFame.length > 3) this.hallOfFame.pop();
    }
    if (this.hallOfFame.length > 1 && this.generation % 4 === 0) this.crossoverHallOfFame();
    const scale = mutationScale * (this.learning.length > 3 && this.brain.fitness < 1 ? 1.35 : 1);
    for (const connection of this.brain.connections) {
      if (this.random.next() < 0.72) connection.weight = clamp(connection.weight + this.random.signed() * 0.22 * scale, -4, 4);
      if (this.random.next() < 0.018 * scale) connection.enabled = !connection.enabled;
    }
    for (const node of this.brain.nodes) {
      if (node.kind !== "input") node.bias = clamp(node.bias + this.random.signed() * 0.04 * scale, -2, 2);
    }
    if (this.random.next() < 0.26 * scale && this.brain.nodes.length < MAX_NEAT_NODES) this.addNeatNode();
    if (this.random.next() < 0.36 * scale) this.addNeatConnection();
    const keys = Object.keys(this.brain.hyperGenes) as (keyof HyperGenes)[];
    for (const key of keys) this.brain.hyperGenes[key] = clamp(this.brain.hyperGenes[key] + this.random.signed() * 0.045 * scale, 0.05, 1.5);
    this.brain.species = this.brain.nodes.length % 3;
    this.trainedNetwork = compileNetwork(this.brain);
  }

  private addNeatNode(): void {
    const enabled = this.brain.connections.filter((connection) => connection.enabled);
    if (enabled.length === 0) return;
    const connection = enabled[this.random.int(enabled.length)];
    connection.enabled = false;
    const id = this.brain.nextNodeId;
    this.brain.nextNodeId += 1;
    const firstOutput = this.brain.nodes.findIndex((node) => node.kind === "output");
    this.brain.nodes.splice(firstOutput, 0, { id, kind: "hidden", bias: this.random.signed() * 0.1, activation: "tanh" });
    this.brain.connections.push({ from: connection.from, to: id, weight: 1, enabled: true, innovation: this.brain.nextInnovation++ });
    this.brain.connections.push({ from: id, to: connection.to, weight: connection.weight, enabled: true, innovation: this.brain.nextInnovation++ });
  }

  private addNeatConnection(): void {
    for (let attempt = 0; attempt < 12; attempt += 1) {
      const fromIndex = this.random.int(this.brain.nodes.length);
      const toIndex = this.random.int(this.brain.nodes.length);
      const from = this.brain.nodes[fromIndex];
      const to = this.brain.nodes[toIndex];
      if (from.kind === "output" || to.kind === "input" || fromIndex >= toIndex) continue;
      let exists = false;
      for (const connection of this.brain.connections) {
        if (connection.from === from.id && connection.to === to.id) {
          exists = true;
          break;
        }
      }
      if (exists) continue;
      this.brain.connections.push({ from: from.id, to: to.id, weight: this.random.signed(), enabled: true, innovation: this.brain.nextInnovation++ });
      return;
    }
  }

  private crossoverHallOfFame(): void {
    const parent = this.hallOfFame[1];
    for (const connection of this.brain.connections) {
      const match = parent.connections.find((candidate) => candidate.innovation === connection.innovation);
      if (match && this.random.next() < 0.35) {
        connection.weight = match.weight;
        connection.enabled = match.enabled;
      }
    }
  }

  forceGenerationForTest(): void {
    this.advanceGeneration();
  }

  getTelemetry(fps = 60): Telemetry {
    let recentDeliveries = 0;
    for (let index = 0; index < this.deliveryWindow.length; index += 1) recentDeliveries += this.deliveryWindow[index];
    const elapsedMinutes = Math.max(1 / 60, Math.min(this.ticks, this.deliveryWindow.length) / 3600);
    const efficiency = recentDeliveries / elapsedMinutes;
    const ablated = !this.grid.ablation.foodTrail || !this.grid.ablation.homeGuide || !this.grid.ablation.alarm || !this.grid.ablation.scoutPulse || !this.grid.ablation.noEntry || !this.grid.ablation.pathIntegration;
    if (!ablated) this.unablatedEfficiency = this.unablatedEfficiency * 0.95 + efficiency * 0.05;
    let predators = 0;
    for (const predator of this.predators) if (predator.active) predators += 1;
    return {
      fps,
      alive: this.alive,
      deaths: this.totalDeaths,
      food: this.totalFood,
      generation: this.generation,
      births: this.totalBirths,
      trainedFood: this.trainedFood,
      randomFood: this.randomFood,
      acoFood: this.acoFood,
      foodThisGeneration: this.foodThisGeneration,
      acoThisGeneration: this.acoThisGeneration,
      randomThisGeneration: this.randomThisGeneration,
      predators,
      meanEnergy: this.meanEnergy,
      efficiency,
      ablatedEfficiencyDrop: ablated ? Math.max(0, this.unablatedEfficiency - efficiency) : 0,
      hiddenNodes: this.brain.nodes.length - INPUT_COUNT - OUTPUT_COUNT,
      synapses: this.brain.connections.reduce((total, connection) => total + (connection.enabled ? 1 : 0), 0),
      brainFitness: this.brain.fitness,
      bottleneck: this.bottleneck,
      storedFood: this.foodBirthProgress,
      ticks: this.ticks,
      turbo: this.turbo,
    };
  }
}
