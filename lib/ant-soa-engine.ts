/**
 * Zero-allocation Structure-of-Arrays colony engine.
 *
 *  - MAX_ANTS = 3000 preallocated typed arrays, Int32Array spatial hash (cellSize = 24)
 *  - three interchangeable policies behind one action interface:
 *      TRAINED_MLP (shared colony brain), HANDCRAFTED_ACO (pheromone-only stigmergy), RANDOM_MLP
 *  - mortality, corpses, necrophoresis, births / colony generations, founder bottleneck
 *  - optional Abyssal Stalker predators with alarm cascade and warden defence
 *  - trophallaxis, tandem running, dead-end / depleted-site sealing
 *
 * Innate reflexes (identical for every policy): 180 degree turn on pickup and on delivery, alarm flight,
 * warden defence, trail damping, tandem following, corpse hauling. Steering, speed and the five gland
 * outputs during FORAGING / RETURNING belong to the policy.
 */
import { PheromoneGrid, PHEROMONE_CELL } from './pheromone-grid';
import { Rng } from './prng-noise';
import { BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS, createRandomGenome, forward, getChampionWeights } from './neat';
import {
  ARC_COMBAT, ARC_TANDEM, ARC_TROPHALLAXIS,
  CASTE_HAULER, CASTE_SCOUT, CASTE_WARDEN,
  CH_ALARM, CH_SCOUT, CHANNEL_COUNT,
  EVENT_FIRST_FOOD, EVENT_FOUNDER, EVENT_GENERATION, EVENT_PREDATOR_SLAIN,
  POLICY_ACO, POLICY_RANDOM, POLICY_TRAINED,
  STATE_ALARMED, STATE_DEAD, STATE_DEFENDING, STATE_FORAGING, STATE_NECROPHORESIS,
  STATE_RETURNING, STATE_TANDEM_FOLLOW, STATE_TRAIL_DAMPING,
} from './types';
import type { AntSoA, ArcBuffer, ArenaMode, FoodSoA, PolicyType, PredatorSoA, SimParams } from './types';

export const MAX_ANTS = 3000;
export const MAX_PREDATORS = 6;
export const MAX_FOOD = 64;
export const MAX_ARCS = 96;
export const HASH_CELL = 24;
export const NEST_RADIUS = 26;
export const ANT_SPEED = 90;
export const PREDATOR_ALARM_RADIUS = 75;
export const PREDATOR_ALARM_BURST = 80;
export const FOOD_PER_BIRTH = 3;
export const BIRTHS_PER_GENERATION = 25;
export const FOUNDER_THRESHOLD = 35;
export const FOUNDER_MUTATION_BOOST = 2.5;
export const FIXED_DT = 1 / 60;

const TWO_PI = Math.PI * 2;
const INV_CELL = 1 / PHEROMONE_CELL;
const TURN_RATE = 5.5;
const SENSOR_DIST = 15;
const SENSOR_ANGLE = 0.62;
const COS_SA = Math.cos(SENSOR_ANGLE);
const SIN_SA = Math.sin(SENSOR_ANGLE);
const LOG_NORM = 1 / Math.log(101);
/** Max gland output in concentration units / s for foodTrail, homeGuide, alarm, scoutPulse, noEntry. */
const GLAND_RATE: readonly number[] = [30, 22, 8, 10, 8];
const FOOD_TAU = 14;
const HOME_TAU = 18;
const ALARM_REFLEX = 0.5;
const WARDEN_ALARM = 0.22;
const PRED_R2 = PREDATOR_ALARM_RADIUS * PREDATOR_ALARM_RADIUS;
const WANDER_KICK = 0.72;

export function createAntSoA(n: number): AntSoA {
  return {
    x: new Float32Array(n), y: new Float32Array(n), vx: new Float32Array(n), vy: new Float32Array(n),
    angle: new Float32Array(n), homeDx: new Float32Array(n), homeDy: new Float32Array(n),
    targetX: new Float32Array(n), targetY: new Float32Array(n), legPhase: new Float32Array(n),
    pheromoneTimer: new Float32Array(n), timeSinceLastVisit: new Float32Array(n), ouNoise: new Float32Array(n),
    energy: new Float32Array(n), age: new Float32Array(n), maxLifespan: new Float32Array(n),
    fitness: new Float32Array(n), memA: new Float32Array(n), memB: new Float32Array(n),
    stateTimer: new Float32Array(n), tripDist: new Float32Array(n), tripTime: new Float32Array(n),
    tripStraight: new Float32Array(n), anchorX: new Float32Array(n), anchorY: new Float32Array(n),
    stallTimer: new Float32Array(n), alarmCooldown: new Float32Array(n),
    state: new Uint8Array(n), caste: new Uint8Array(n), policyType: new Uint8Array(n),
    carryingFood: new Uint8Array(n), active: new Uint8Array(n), cohort: new Uint8Array(n), bump: new Uint8Array(n),
    generation: new Uint16Array(n), link: new Int16Array(n),
  };
}

function angDiff(target: number, current: number): number {
  let d = target - current;
  d -= TWO_PI * Math.round(d / TWO_PI);
  return d;
}

export class ColonySim {
  readonly width: number;
  readonly height: number;
  readonly grid: PheromoneGrid;
  readonly rng: Rng;
  readonly ants: AntSoA = createAntSoA(MAX_ANTS);
  readonly pred: PredatorSoA = {
    x: new Float32Array(MAX_PREDATORS), y: new Float32Array(MAX_PREDATORS), angle: new Float32Array(MAX_PREDATORS),
    hp: new Float32Array(MAX_PREDATORS), cooldown: new Float32Array(MAX_PREDATORS), phase: new Float32Array(MAX_PREDATORS),
    target: new Int16Array(MAX_PREDATORS), active: new Uint8Array(MAX_PREDATORS),
  };
  readonly food: FoodSoA = {
    x: new Float32Array(MAX_FOOD), y: new Float32Array(MAX_FOOD), amount: new Float32Array(MAX_FOOD),
    initial: new Float32Array(MAX_FOOD), radius: new Float32Array(MAX_FOOD), depletedTimer: new Float32Array(MAX_FOOD),
    variant: new Uint8Array(MAX_FOOD), active: new Uint8Array(MAX_FOOD),
  };
  readonly arcs: ArcBuffer = {
    x1: new Float32Array(MAX_ARCS), y1: new Float32Array(MAX_ARCS), x2: new Float32Array(MAX_ARCS),
    y2: new Float32Array(MAX_ARCS), kind: new Uint8Array(MAX_ARCS), ttl: new Uint8Array(MAX_ARCS),
  };

  /** High-water mark of used ant slots (live ants + corpses + holes). */
  count = 0;
  alive = 0;
  readonly cohortAlive = new Int32Array(2);
  private readonly freeStack = new Int16Array(MAX_ANTS);
  private freeTop = 0;

  readonly nestX = new Float32Array(2);
  readonly nestY = new Float32Array(2);
  readonly graveX = new Float32Array(2);
  readonly graveY = new Float32Array(2);
  nestCount = 1;
  foodCount = 0;
  predatorCount = 0;
  private arcHead = 0;

  readonly hashCols: number;
  readonly hashRows: number;
  readonly hashHead: Int32Array;
  readonly hashNext = new Int32Array(MAX_ANTS);

  /** Shared colony brain (TRAINED_MLP) and the random-network floor (RANDOM_MLP). */
  readonly brainA: Float32Array = getChampionWeights();
  readonly brainB: Float32Array = createRandomGenome(new Rng(99), 0.7);
  policyA: PolicyType = POLICY_TRAINED;
  policyB: PolicyType = POLICY_RANDOM;
  arena: ArenaMode = 'trained';

  params: SimParams = { maxPop: 1600, atpBurn: 1, lifespan: 200, mortality: true, births: true, founderWaves: true };
  /** Path-integration ablation switch. */
  pathIntegration = true;
  /** Adaptive scaler / headless: run reaction-diffusion every 2nd tick with dt * 2. */
  halfRateDiffusion = false;

  tick = 0;
  time = 0;
  totalDeaths = 0;
  births = 0;
  colonyGeneration = 1;
  storedFood = 0;
  readonly delivered = new Int32Array(2);
  readonly broodCredit = new Float32Array(2);
  pickups = 0;
  effSum = 0;
  wallHits = 0;
  antTicks = 0;
  loops = 0;
  troph = 0;
  defenseHits = 0;
  starved = 0;
  homing = 0;
  predKills = 0;
  /** Bitmask of EVENT_* flags; the UI reads and clears it. */
  events = 0;
  founderBoost = 1;
  private founderCooldown = 0;

  private readonly inp = new Float32Array(BRAIN_INPUTS);
  private readonly hid = new Float32Array(BRAIN_HIDDEN);
  private readonly out = new Float32Array(BRAIN_OUTPUTS);

  constructor(width: number, height: number, seed: number) {
    this.width = width;
    this.height = height;
    this.grid = new PheromoneGrid(width, height);
    this.rng = new Rng(seed);
    this.hashCols = Math.ceil(width / HASH_CELL) + 1;
    this.hashRows = Math.ceil(height / HASH_CELL) + 1;
    this.hashHead = new Int32Array(this.hashCols * this.hashRows).fill(-1);
    this.nestX[0] = width / 2;
    this.nestY[0] = height / 2;
  }

  /* ------------------------------------------------------------------ */
  /* World management                                                    */
  /* ------------------------------------------------------------------ */

  clearWorld(): void {
    this.ants.active.fill(0);
    this.count = 0;
    this.alive = 0;
    this.freeTop = 0;
    this.cohortAlive.fill(0);
    this.food.active.fill(0);
    this.foodCount = 0;
    this.pred.active.fill(0);
    this.predatorCount = 0;
    this.arcs.ttl.fill(0);
    this.grid.clearPheromones();
    this.grid.clearWalls();
    this.resetStats();
  }

  resetStats(): void {
    this.tick = 0;
    this.time = 0;
    this.totalDeaths = 0;
    this.births = 0;
    this.colonyGeneration = 1;
    this.storedFood = 0;
    this.delivered.fill(0);
    this.broodCredit.fill(0);
    this.pickups = 0;
    this.effSum = 0;
    this.wallHits = 0;
    this.antTicks = 0;
    this.loops = 0;
    this.troph = 0;
    this.defenseHits = 0;
    this.starved = 0;
    this.homing = 0;
    this.predKills = 0;
    this.events = 0;
    this.founderBoost = 1;
    this.founderCooldown = 0;
  }

  setArena(mode: ArenaMode): void {
    this.arena = mode;
    this.policyA = mode === 'aco' ? POLICY_ACO : POLICY_TRAINED;
    this.policyB = mode === 'ab-aco' ? POLICY_ACO : POLICY_RANDOM;
  }

  setBrain(weights: Float32Array): void {
    this.brainA.set(weights);
  }

  setNest(c: number, x: number, y: number): void {
    this.nestX[c] = x;
    this.nestY[c] = y;
    this.graveX[c] = x;
    this.graveY[c] = y + 70;
    // keep the crater itself free of basalt
    this.grid.setWallDisc(x, y, NEST_RADIUS + 10, false);
  }

  /** Graveyards sit ~84px from the nest, in the open direction farthest from every food site. */
  placeGraveyards(): void {
    const f = this.food;
    for (let c = 0; c < this.nestCount; c++) {
      let best = -1;
      let bx = this.nestX[c];
      let by = this.nestY[c] + 60;
      for (let k = 0; k < 12; k++) {
        const ang = (k * Math.PI) / 6;
        const gx = this.nestX[c] + Math.cos(ang) * 84;
        const gy = this.nestY[c] + Math.sin(ang) * 84;
        if (gx < 20 || gy < 20 || gx > this.width - 20 || gy > this.height - 20) continue;
        if (this.grid.isWallAt(gx, gy)) continue;
        let score = 1e9;
        for (let i = 0; i < this.foodCount; i++) {
          if (f.active[i] === 0) continue;
          const dx = f.x[i] - gx;
          const dy = f.y[i] - gy;
          const d = dx * dx + dy * dy;
          if (d < score) score = d;
        }
        if (score > best) {
          best = score;
          bx = gx;
          by = gy;
        }
      }
      this.graveX[c] = bx;
      this.graveY[c] = by;
    }
  }

  populate(n: number): void {
    for (let k = 0; k < n; k++) this.spawnAnt(this.nestCount === 2 ? k & 1 : 0, false);
  }

  /** Spawns one ant at its cohort's nest. Returns the slot or -1 when the SoA is full. */
  spawnAnt(c: number, newborn: boolean): number {
    let i: number;
    if (this.freeTop > 0) i = this.freeStack[--this.freeTop];
    else if (this.count < MAX_ANTS) i = this.count++;
    else return -1;
    const a = this.ants;
    const rng = this.rng;
    const r = Math.sqrt(rng.next()) * NEST_RADIUS * 0.8;
    const th = rng.next() * TWO_PI;
    const px = this.nestX[c] + Math.cos(th) * r;
    const py = this.nestY[c] + Math.sin(th) * r;
    a.x[i] = px;
    a.y[i] = py;
    a.vx[i] = 0;
    a.vy[i] = 0;
    a.angle[i] = rng.next() * TWO_PI;
    a.homeDx[i] = this.nestX[c] - px;
    a.homeDy[i] = this.nestY[c] - py;
    a.targetX[i] = px;
    a.targetY[i] = py;
    a.legPhase[i] = rng.next() * 8;
    a.pheromoneTimer[i] = 0;
    a.timeSinceLastVisit[i] = 0;
    a.ouNoise[i] = 0;
    a.energy[i] = newborn ? 100 : 60 + rng.next() * 40;
    a.maxLifespan[i] = this.params.lifespan * (0.7 + rng.next() * 0.6);
    a.age[i] = newborn ? 0 : rng.next() * 0.45 * this.params.lifespan;
    a.fitness[i] = 0;
    a.memA[i] = 0;
    a.memB[i] = 0;
    a.stateTimer[i] = 0;
    a.tripDist[i] = 0;
    a.tripTime[i] = 0;
    a.tripStraight[i] = 0;
    a.anchorX[i] = px;
    a.anchorY[i] = py;
    a.stallTimer[i] = 1 + rng.next() * 2.5;
    a.alarmCooldown[i] = 0;
    a.state[i] = STATE_FORAGING;
    const cr = rng.next();
    a.caste[i] = cr < 0.25 ? CASTE_SCOUT : cr < 0.85 ? CASTE_HAULER : CASTE_WARDEN;
    a.policyType[i] = c === 0 ? this.policyA : this.policyB;
    a.carryingFood[i] = 0;
    a.active[i] = 1;
    a.cohort[i] = c;
    a.bump[i] = 0;
    a.generation[i] = this.colonyGeneration;
    a.link[i] = -1;
    this.alive++;
    this.cohortAlive[c]++;
    return i;
  }

  addFood(x: number, y: number, amount: number): number {
    const f = this.food;
    for (let i = 0; i < this.foodCount; i++) {
      if (f.active[i] === 0 || f.amount[i] < 1) continue;
      const dx = f.x[i] - x;
      const dy = f.y[i] - y;
      if (dx * dx + dy * dy < 18 * 18) {
        f.amount[i] += amount;
        if (f.amount[i] > f.initial[i]) f.initial[i] = f.amount[i];
        f.radius[i] = 4 + 1.3 * Math.sqrt(f.amount[i]);
        return i;
      }
    }
    let slot = -1;
    for (let i = 0; i < this.foodCount; i++) {
      if (f.active[i] === 0) {
        slot = i;
        break;
      }
    }
    if (slot < 0) {
      if (this.foodCount >= MAX_FOOD) return -1;
      slot = this.foodCount++;
    }
    f.x[slot] = x;
    f.y[slot] = y;
    f.amount[slot] = amount;
    f.initial[slot] = amount;
    f.radius[slot] = 4 + 1.3 * Math.sqrt(amount);
    f.depletedTimer[slot] = 0;
    f.variant[slot] = ((x * 13 + y * 7) | 0) & 7;
    f.active[slot] = 1;
    return slot;
  }

  totalFoodRemaining(): number {
    let s = 0;
    for (let i = 0; i < this.foodCount; i++) if (this.food.active[i] !== 0) s += this.food.amount[i];
    return s;
  }

  paintWall(x: number, y: number, r: number, solid: boolean): number {
    // never bury a nest
    if (solid) {
      for (let c = 0; c < this.nestCount; c++) {
        const dx = this.nestX[c] - x;
        const dy = this.nestY[c] - y;
        const lim = NEST_RADIUS + r + 6;
        if (dx * dx + dy * dy < lim * lim) return 0;
      }
    }
    return this.grid.setWallDisc(x, y, r, solid);
  }

  /** Eraser: pheromones, walls, food sites and predators inside the disc. Returns wall cells changed. */
  eraseAt(x: number, y: number, r: number): number {
    this.grid.eraseDisc(x, y, r);
    const f = this.food;
    for (let i = 0; i < this.foodCount; i++) {
      if (f.active[i] === 0) continue;
      const dx = f.x[i] - x;
      const dy = f.y[i] - y;
      if (dx * dx + dy * dy < r * r) f.active[i] = 0;
    }
    const p = this.pred;
    for (let k = 0; k < MAX_PREDATORS; k++) {
      if (p.active[k] === 0) continue;
      const dx = p.x[k] - x;
      const dy = p.y[k] - y;
      if (dx * dx + dy * dy < r * r) {
        p.active[k] = 0;
        this.predatorCount--;
      }
    }
    return this.grid.setWallDisc(x, y, r, false);
  }

  spawnPredator(x: number, y: number): boolean {
    if (this.predatorCount >= MAX_PREDATORS) return false;
    if (this.grid.isWallAt(x, y)) return false;
    const p = this.pred;
    for (let k = 0; k < MAX_PREDATORS; k++) {
      if (p.active[k] !== 0) continue;
      p.x[k] = x;
      p.y[k] = y;
      p.angle[k] = this.rng.next() * TWO_PI;
      p.hp[k] = 100;
      p.cooldown[k] = 0.5;
      p.phase[k] = 0;
      p.target[k] = -1;
      p.active[k] = 1;
      this.predatorCount++;
      return true;
    }
    return false;
  }

  /** Brings the live predator count to n by spawning far from nests or culling. */
  setPredatorCount(n: number): void {
    const p = this.pred;
    for (let k = MAX_PREDATORS - 1; k >= 0 && this.predatorCount > n; k--) {
      if (p.active[k] !== 0) {
        p.active[k] = 0;
        this.predatorCount--;
      }
    }
    let guard = 0;
    while (this.predatorCount < n && guard++ < 200) {
      const x = 30 + this.rng.next() * (this.width - 60);
      const y = 30 + this.rng.next() * (this.height - 60);
      let ok = true;
      for (let c = 0; c < this.nestCount; c++) {
        const dx = this.nestX[c] - x;
        const dy = this.nestY[c] - y;
        if (dx * dx + dy * dy < 170 * 170) ok = false;
      }
      if (ok) this.spawnPredator(x, y);
    }
  }

  private pushArc(x1: number, y1: number, x2: number, y2: number, kind: number): void {
    const k = this.arcHead;
    this.arcHead = (k + 1) % MAX_ARCS;
    const arcs = this.arcs;
    arcs.x1[k] = x1;
    arcs.y1[k] = y1;
    arcs.x2[k] = x2;
    arcs.y2[k] = y2;
    arcs.kind[k] = kind;
    arcs.ttl[k] = 3;
  }

  private kill(i: number, starvedDeath: boolean): void {
    const a = this.ants;
    if (a.carryingFood[i] !== 0) {
      a.carryingFood[i] = 0;
      this.addFood(a.x[i], a.y[i], 1);
    }
    const l = a.link[i];
    if (l >= 0 && a.state[i] === STATE_NECROPHORESIS) a.link[l] = -1;
    a.link[i] = -1;
    a.state[i] = STATE_DEAD;
    a.age[i] = 0;
    a.vx[i] = 0;
    a.vy[i] = 0;
    this.alive--;
    this.cohortAlive[a.cohort[i]]--;
    this.totalDeaths++;
    if (starvedDeath) this.starved++;
  }

  private freeSlot(i: number): void {
    const a = this.ants;
    const l = a.link[i];
    if (l >= 0) a.link[l] = -1;
    a.active[i] = 0;
    this.freeStack[this.freeTop++] = i;
  }

  private stepCorpse(i: number, dt: number, tick: number): void {
    const a = this.ants;
    a.age[i] += dt;
    const l = a.link[i];
    if (l >= 0) {
      if (a.active[l] === 0 || a.state[l] === STATE_DEAD) {
        a.link[i] = -1;
      } else {
        const ang = a.angle[l];
        a.x[i] = a.x[l] - Math.cos(ang) * 5;
        a.y[i] = a.y[l] - Math.sin(ang) * 5;
        a.angle[i] = ang + 1.2;
      }
    } else if (l === -1 && ((tick + i) & 15) === 0) {
      // weak necrotic alarm
      this.grid.deposit(CH_ALARM, a.x[i], a.y[i], 0.6);
    }
    if (a.age[i] > (l === -2 ? 14 : 50)) this.freeSlot(i);
  }

  private rebuildHash(): void {
    const head = this.hashHead;
    const next = this.hashNext;
    const a = this.ants;
    const cols = this.hashCols;
    head.fill(-1);
    for (let i = 0; i < this.count; i++) {
      if (a.active[i] === 0) continue;
      const c = ((a.y[i] / HASH_CELL) | 0) * cols + ((a.x[i] / HASH_CELL) | 0);
      next[i] = head[c];
      head[c] = i;
    }
  }

  /** Neighbour interactions via the spatial hash: necrophoresis pickup, trophallaxis, tandem recruitment. */
  private interactions(): void {
    const a = this.ants;
    const head = this.hashHead;
    const next = this.hashNext;
    const hc = this.hashCols;
    const hr = this.hashRows;
    const n = this.count;
    for (let i = 0; i < n; i++) {
      if (a.active[i] === 0) continue;
      const st = a.state[i];
      let mode = -1;
      if (st === STATE_DEAD) {
        if (a.link[i] === -1) mode = 0;
      } else if (a.energy[i] < 30) mode = 1;
      else if (a.caste[i] === CASTE_SCOUT && a.carryingFood[i] !== 0 && st === STATE_RETURNING) mode = 2;
      if (mode < 0) continue;
      const px = a.x[i];
      const py = a.y[i];
      const cx = (px / HASH_CELL) | 0;
      const cy = (py / HASH_CELL) | 0;
      const co = a.cohort[i];
      let recruits = 0;
      let done = false;
      for (let oy = -1; oy <= 1 && !done; oy++) {
        const hy = cy + oy;
        if (hy < 0 || hy >= hr) continue;
        for (let ox = -1; ox <= 1 && !done; ox++) {
          const hx = cx + ox;
          if (hx < 0 || hx >= hc) continue;
          let j = head[hy * hc + hx];
          let guard = 0;
          while (j >= 0 && guard++ < 24) {
            const sj = a.state[j];
            if (j !== i && a.active[j] !== 0 && a.cohort[j] === co && sj !== STATE_DEAD) {
              const dx = a.x[j] - px;
              const dy = a.y[j] - py;
              const d2 = dx * dx + dy * dy;
              if (mode === 0) {
                if (a.caste[j] === CASTE_WARDEN && a.carryingFood[j] === 0 && a.link[j] < 0 &&
                  (sj === STATE_FORAGING || sj === STATE_DEFENDING) && d2 < 256) {
                  a.link[i] = j;
                  a.link[j] = i;
                  a.state[j] = STATE_NECROPHORESIS;
                  done = true;
                  break;
                }
              } else if (mode === 1) {
                if (a.energy[j] > 60 && d2 < 144) {
                  const give = Math.min(22, (a.energy[j] - a.energy[i]) * 0.5);
                  a.energy[j] -= give;
                  a.energy[i] += give;
                  a.fitness[j] += 0.5;
                  this.troph++;
                  this.pushArc(a.x[j], a.y[j], px, py, ARC_TROPHALLAXIS);
                  done = true;
                  break;
                }
              } else if (sj === STATE_FORAGING && a.carryingFood[j] === 0 && d2 < 324) {
                a.targetX[j] = a.targetX[i];
                a.targetY[j] = a.targetY[i];
                a.state[j] = STATE_TANDEM_FOLLOW;
                a.stateTimer[j] = 4;
                this.grid.deposit(CH_SCOUT, px, py, 25);
                this.pushArc(px, py, a.x[j], a.y[j], ARC_TANDEM);
                if (++recruits >= 2) {
                  done = true;
                  break;
                }
              }
            }
            j = next[j];
          }
        }
      }
    }
  }

  private slayPredator(k: number): void {
    const p = this.pred;
    p.active[k] = 0;
    this.predatorCount--;
    this.events |= EVENT_PREDATOR_SLAIN;
    this.addFood(p.x[k], p.y[k], 18);
  }

  private stepPredators(dt: number): void {
    const p = this.pred;
    const a = this.ants;
    const grid = this.grid;
    const tick = this.tick;
    for (let k = 0; k < MAX_PREDATORS; k++) {
      if (p.active[k] === 0) continue;
      p.cooldown[k] -= dt;
      p.phase[k] += dt * 7;
      const px = p.x[k];
      const py = p.y[k];
      if (((tick + k * 3) & 15) === 0) {
        let best = -1;
        let bd = 170 * 170;
        for (let i = 0; i < this.count; i++) {
          if (a.active[i] === 0 || a.state[i] === STATE_DEAD) continue;
          const dx = a.x[i] - px;
          const dy = a.y[i] - py;
          const d2 = dx * dx + dy * dy;
          if (d2 < bd) {
            bd = d2;
            best = i;
          }
        }
        p.target[k] = best;
      }
      let ang = p.angle[k];
      const t = p.target[k];
      if (t >= 0 && a.active[t] !== 0 && a.state[t] !== STATE_DEAD) {
        const dx = a.x[t] - px;
        const dy = a.y[t] - py;
        let turn = angDiff(Math.atan2(dy, dx), ang);
        const lim = 3.2 * dt;
        if (turn > lim) turn = lim;
        else if (turn < -lim) turn = -lim;
        ang += turn;
        if (dx * dx + dy * dy < 14 * 14 && p.cooldown[k] <= 0) {
          this.pushArc(px, py, a.x[t], a.y[t], ARC_COMBAT);
          this.kill(t, false);
          this.predKills++;
          p.cooldown[k] = 0.9;
          p.hp[k] = Math.min(100, p.hp[k] + 3);
          p.target[k] = -1;
        }
      } else {
        ang += (this.rng.next() - 0.5) * 0.25;
      }
      const nx = px + Math.cos(ang) * 46 * dt;
      const ny = py + Math.sin(ang) * 46 * dt;
      if (nx < 12 || ny < 12 || nx > this.width - 12 || ny > this.height - 12 || grid.isWallAt(nx, ny)) {
        ang += 0.9;
      } else {
        p.x[k] = nx;
        p.y[k] = ny;
      }
      p.angle[k] = ang;
    }
  }

  private ray(px: number, py: number, dx: number, dy: number): number {
    const g = this.grid;
    const cols = g.cols;
    const rows = g.rows;
    const wall = g.wall;
    for (let k = 1; k <= 4; k++) {
      const X = px + dx * 6 * k;
      const Y = py + dy * 6 * k;
      if (X < 0 || Y < 0) return 1.25 - k * 0.25;
      const gx = (X * INV_CELL) | 0;
      const gy = (Y * INV_CELL) | 0;
      if (gx >= cols || gy >= rows || wall[gy * cols + gx] !== 0) return 1.25 - k * 0.25;
    }
    return 0;
  }

  /**
   * Hand-crafted ACO baseline: pure stigmergy, no path integration.
   * Foragers climb foodTrail, carriers climb homeGuide (bilateral sensors), avoid noEntry + alarm,
   * are drawn to scoutPulse, steer off walls; foragers secrete homeGuide, carriers secrete foodTrail.
   */
  private aco(i: number, carrying: boolean): void {
    const inp = this.inp;
    const out = this.out;
    const follow = carrying ? 1 : 0;
    let steer = 2.6 * inp[follow];
    steer -= 1.6 * inp[4];
    steer -= 1.2 * inp[2];
    if (!carrying) steer += 0.9 * inp[3];
    steer += 2.2 * (inp[10] - inp[12]);
    const side = this.ants.memA[i] >= 0 ? 1 : -1;
    if (inp[11] > 0.4) steer += side * 2 * inp[11];
    out[0] = steer > 1 ? 1 : steer < -1 ? -1 : steer;
    out[1] = 1;
    out[2] = carrying ? 1 : 0;
    out[3] = carrying ? 0 : 1;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    // memA remembers a preferred side for head-on walls; flips rarely
    out[7] = inp[10] > inp[12] + 0.2 ? 1 : inp[12] > inp[10] + 0.2 ? -1 : side;
    out[8] = 0;
  }

  /* ------------------------------------------------------------------ */
  /* Fixed-step update                                                   */
  /* ------------------------------------------------------------------ */

  step(dt: number): void {
    const a = this.ants;
    const x = a.x, y = a.y, angle = a.angle, homeDx = a.homeDx, homeDy = a.homeDy;
    const state = a.state, carry = a.carryingFood, active = a.active, caste = a.caste;
    const policy = a.policyType, cohort = a.cohort, energy = a.energy, stateTimer = a.stateTimer, link = a.link;
    const grid = this.grid;
    const cols = grid.cols;
    const wall = grid.wall;
    const mask = grid.mask;
    const chActive = grid.active;
    const cur = grid.cur;
    const inp = this.inp, hid = this.hid, out = this.out;
    const W = this.width, H = this.height;
    const rng = this.rng;
    const pred = this.pred;
    const food = this.food;
    const tick = ++this.tick;
    this.time += dt;

    if ((tick & 7) === 0) {
      this.rebuildHash();
      this.interactions();
    }
    if (this.predatorCount > 0) this.stepPredators(dt);
    for (let f = 0; f < this.foodCount; f++) {
      if (food.active[f] !== 0 && food.amount[f] < 1) {
        food.depletedTimer[f] -= dt;
        if (food.depletedTimer[f] <= 0) food.active[f] = 0;
      }
    }

    const atp = this.params.atpBurn;
    const mortal = this.params.mortality;
    const pathInt = this.pathIntegration;
    const hasPred = this.predatorCount > 0;
    const noWalls = grid.wallCount === 0;
    const nestR2 = NEST_RADIUS * NEST_RADIUS;
    const n = this.count;

    for (let i = 0; i < n; i++) {
      if (active[i] === 0) continue;
      let st = state[i];
      if (st === STATE_DEAD) {
        this.stepCorpse(i, dt, tick);
        continue;
      }
      const px = x[i];
      const py = y[i];
      let ang = angle[i];
      let carrying = carry[i] !== 0;
      const c = cohort[i];
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      const lx = ca * COS_SA + sa * SIN_SA;
      const ly = sa * COS_SA - ca * SIN_SA;
      const rx = ca * COS_SA - sa * SIN_SA;
      const ry = sa * COS_SA + ca * SIN_SA;

      /* ---- olfaction: Weber-Fechner S = log(1 + C), bilateral gradient + centre ---- */
      const iL = grid.cellIndex(px + lx * SENSOR_DIST, py + ly * SENSOR_DIST);
      const iR = grid.cellIndex(px + rx * SENSOR_DIST, py + ry * SENSOR_DIST);
      const iC = grid.cellIndex(px + ca * SENSOR_DIST * 0.75, py + sa * SENSOR_DIST * 0.75);
      for (let ch = 0; ch < CHANNEL_COUNT; ch++) {
        if (mask[ch] === 0 || chActive[ch] === 0) {
          inp[ch] = 0;
          inp[5 + ch] = 0;
          continue;
        }
        const g = cur[ch];
        const sL = iL >= 0 ? Math.log(1 + g[iL]) : 0;
        const sR = iR >= 0 ? Math.log(1 + g[iR]) : 0;
        inp[ch] = sR - sL;
        inp[5 + ch] = iC >= 0 ? Math.log(1 + g[iC]) * LOG_NORM : 0;
      }

      /* ---- wall raycasts (left, centre, right) ---- */
      if (noWalls && px > 28 && py > 28 && px < W - 28 && py < H - 28) {
        inp[10] = 0;
        inp[11] = 0;
        inp[12] = 0;
      } else {
        inp[10] = this.ray(px, py, lx, ly);
        inp[11] = this.ray(px, py, ca, sa);
        inp[12] = this.ray(px, py, rx, ry);
      }

      /* ---- path integration (egocentric, distance-saturated) ---- */
      const hx = homeDx[i];
      const hy = homeDy[i];
      const hd = Math.sqrt(hx * hx + hy * hy);
      if (pathInt && hd > 1e-3) {
        const sc = 1 / (hd + 60);
        inp[13] = (hx * ca + hy * sa) * sc;
        inp[14] = (-hx * sa + hy * ca) * sc;
      } else {
        inp[13] = 0;
        inp[14] = 0;
      }
      inp[15] = carrying ? 1 : 0;
      inp[16] = energy[i] * 0.01;
      inp[17] = a.memA[i];
      inp[18] = a.memB[i];

      /* ---- policy ---- */
      const pol = policy[i];
      if (pol === POLICY_ACO) this.aco(i, carrying);
      else forward(pol === POLICY_TRAINED ? this.brainA : this.brainB, inp, hid, out);
      let steer = out[0];
      let speedMod = out[1];
      let e0 = out[2];
      const e1 = out[3];
      const e2 = out[4];
      const e3 = out[5];
      let e4 = out[6];
      a.memA[i] = out[7];
      a.memB[i] = out[8];

      /* ---- state machine + innate reflexes ---- */
      if (st === STATE_NECROPHORESIS) {
        if (link[i] < 0) st = STATE_FORAGING;
      } else if (st >= STATE_ALARMED) {
        const t = stateTimer[i] - dt;
        stateTimer[i] = t;
        if (t <= 0) st = carrying ? STATE_RETURNING : STATE_FORAGING;
      } else {
        st = carrying ? STATE_RETURNING : STATE_FORAGING;
      }
      const alarmG = inp[2];
      const alarmC = inp[7];
      const isWarden = caste[i] === CASTE_WARDEN;
      if (st <= STATE_RETURNING || st === STATE_TANDEM_FOLLOW) {
        if (isWarden) {
          if (!carrying && alarmC > WARDEN_ALARM) {
            st = STATE_DEFENDING;
            stateTimer[i] = 0.6;
          }
        } else if (alarmC > ALARM_REFLEX) {
          st = STATE_ALARMED;
          stateTimer[i] = 0.8;
        }
      }
      let nearP = -1;
      let nearD2 = 1e12;
      if (hasPred) {
        for (let k = 0; k < MAX_PREDATORS; k++) {
          if (pred.active[k] === 0) continue;
          const dx = pred.x[k] - px;
          const dy = pred.y[k] - py;
          const d2 = dx * dx + dy * dy;
          if (d2 < nearD2) {
            nearD2 = d2;
            nearP = k;
          }
        }
        if (nearP >= 0 && nearD2 < PRED_R2) {
          // danger broadcast: explosive crimson burst
          if (a.alarmCooldown[i] <= 0) {
            grid.deposit(CH_ALARM, px, py, PREDATOR_ALARM_BURST);
            a.alarmCooldown[i] = 1.2 + rng.next() * 0.6;
          }
          if (isWarden) {
            if (st !== STATE_NECROPHORESIS) {
              st = STATE_DEFENDING;
              stateTimer[i] = 1;
            }
          } else if (st !== STATE_NECROPHORESIS) {
            st = STATE_ALARMED;
            stateTimer[i] = 1;
          }
        }
      }
      if (a.alarmCooldown[i] > 0) a.alarmCooldown[i] -= dt;

      if (st === STATE_ALARMED) {
        // flee down the alarm gradient, toward homeGuide
        steer = -3 * alarmG + 2.5 * inp[1] + 0.4 * steer;
        if (nearP >= 0 && nearD2 < 110 * 110) {
          const dx = pred.x[nearP] - px;
          const dy = pred.y[nearP] - py;
          const d = Math.sqrt(nearD2) + 1e-3;
          const lat = (-dx * sa + dy * ca) / d;
          const fwd = (dx * ca + dy * sa) / d;
          steer -= 2.5 * lat + (fwd > 0 ? (lat >= 0 ? 1.5 : -1.5) : 0);
        }
        speedMod = 1;
      } else if (st === STATE_DEFENDING) {
        // wardens chemotax up the alarm gradient and swarm the predator
        steer = 3 * alarmG + 0.3 * steer;
        speedMod = 1;
        if (nearP >= 0 && nearD2 < 120 * 120) {
          const dx = pred.x[nearP] - px;
          const dy = pred.y[nearP] - py;
          const d = Math.sqrt(nearD2) + 1e-3;
          const lat = (-dx * sa + dy * ca) / d;
          const fwd = (dx * ca + dy * sa) / d;
          steer = 2.8 * lat + (fwd < 0 ? (lat >= 0 ? 1.5 : -1.5) : 0);
          if (nearD2 < 16 * 16) {
            pred.hp[nearP] -= 16 * dt;
            energy[i] -= 4 * dt * atp;
            this.defenseHits += dt;
            a.fitness[i] += dt;
            speedMod = 0.15;
            if (((tick + i) & 7) === 0) this.pushArc(px, py, pred.x[nearP], pred.y[nearP], ARC_COMBAT);
            if (pred.hp[nearP] <= 0) this.slayPredator(nearP);
          }
        }
      } else if (st === STATE_TRAIL_DAMPING) {
        steer *= 0.3;
        speedMod = 1;
        e4 = 1;
        e0 = 0;
      } else if (st === STATE_TANDEM_FOLLOW) {
        const dx = a.targetX[i] - px;
        const dy = a.targetY[i] - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < 18 * 18 || carrying) {
          st = carrying ? STATE_RETURNING : STATE_FORAGING;
          stateTimer[i] = 0;
        } else {
          const d = Math.sqrt(d2);
          const lat = (-dx * sa + dy * ca) / d;
          const fwd = (dx * ca + dy * sa) / d;
          steer = 0.3 * steer + 2.2 * lat + (fwd < 0 ? (lat >= 0 ? 1 : -1) : 0);
          if (speedMod < 0.8) speedMod = 0.8;
        }
      } else if (st === STATE_NECROPHORESIS) {
        const dx = this.graveX[c] - px;
        const dy = this.graveY[c] - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < 14 * 14) {
          const corpse = link[i];
          if (corpse >= 0) link[corpse] = -2;
          link[i] = -1;
          st = STATE_FORAGING;
          a.fitness[i] += 0.5;
        } else {
          const d = Math.sqrt(d2);
          const lat = (-dx * sa + dy * ca) / d;
          const fwd = (dx * ca + dy * sa) / d;
          steer = 2.4 * lat + (fwd < 0 ? (lat >= 0 ? 1 : -1) : 0) + 1.5 * (inp[10] - inp[12]);
          speedMod = 0.8;
        }
      }

      /* ---- actuation ---- */
      const isScout = caste[i] === CASTE_SCOUT;
      let ou = a.ouNoise[i];
      ou += -1.6 * ou * dt + (rng.next() - 0.5) * (isScout ? WANDER_KICK * 1.35 : WANDER_KICK);
      a.ouNoise[i] = ou;
      if (steer > 1) steer = 1;
      else if (steer < -1) steer = -1;
      ang += (steer * TURN_RATE + ou) * dt;
      if (speedMod < 0) speedMod = 0;
      else if (speedMod > 1) speedMod = 1;
      const speedFrac = 0.2 + 0.8 * speedMod;
      const sp = ANT_SPEED * speedFrac * (isScout ? 1.15 : isWarden ? 0.92 : 1) * (carrying ? 0.9 : 1);
      let nx = px + ca * sp * dt;
      let ny = py + sa * sp * dt;
      const hereWall = wall[((py * INV_CELL) | 0) * cols + ((px * INV_CELL) | 0)] !== 0;
      if (!hereWall) {
        const oob = nx < 2 || ny < 2 || nx > W - 2 || ny > H - 2;
        if (oob || wall[((ny * INV_CELL) | 0) * cols + ((nx * INV_CELL) | 0)] !== 0) {
          const okX = nx >= 2 && nx <= W - 2 && wall[((py * INV_CELL) | 0) * cols + ((nx * INV_CELL) | 0)] === 0;
          const okY = ny >= 2 && ny <= H - 2 && wall[((ny * INV_CELL) | 0) * cols + ((px * INV_CELL) | 0)] === 0;
          if (okX) {
            ny = py;
            ang += angDiff(ca >= 0 ? 0 : Math.PI, ang) * 0.22;
          } else if (okY) {
            nx = px;
            ang += angDiff(sa >= 0 ? Math.PI * 0.5 : -Math.PI * 0.5, ang) * 0.22;
          } else {
            nx = px;
            ny = py;
            ang += 0.5 + (rng.next() - 0.5) * 1.6;
          }
          this.wallHits++;
          if (a.bump[i] < 250) a.bump[i]++;
        }
      } else {
        if (nx < 2) nx = 2;
        else if (nx > W - 2) nx = W - 2;
        if (ny < 2) ny = 2;
        else if (ny > H - 2) ny = H - 2;
      }
      if (ang > 100 || ang < -100) ang -= TWO_PI * Math.round(ang / TWO_PI);
      const ddx = nx - px;
      const ddy = ny - py;
      const moved = Math.sqrt(ddx * ddx + ddy * ddy);
      x[i] = nx;
      y[i] = ny;
      a.vx[i] = ddx / dt;
      a.vy[i] = ddy / dt;
      const nhx = hx - ddx;
      const nhy = hy - ddy;
      homeDx[i] = nhx;
      homeDy[i] = nhy;
      a.legPhase[i] += moved * 0.33;
      this.antTicks++;

      /* ---- metabolism & mortality ---- */
      a.age[i] += dt;
      a.timeSinceLastVisit[i] += dt;
      if (mortal) {
        const en = energy[i] - atp * (0.3 + 0.5 * speedFrac + (carrying ? 0.2 : 0)) * dt;
        energy[i] = en;
        if (en <= 0 || a.age[i] >= a.maxLifespan[i]) {
          angle[i] = ang;
          this.kill(i, en <= 0);
          continue;
        }
      }

      const nd2 = nhx * nhx + nhy * nhy;
      if (carrying) {
        a.tripDist[i] += moved;
        a.tripTime[i] += dt;
        this.homing += hd - Math.sqrt(nd2);
      }

      /* ---- nest: refuel, deliver ---- */
      if (nd2 < nestR2) {
        energy[i] = 100;
        a.timeSinceLastVisit[i] = 0;
        if (carrying) {
          carry[i] = 0;
          carrying = false;
          if (this.storedFood === 0) this.events |= EVENT_FIRST_FOOD;
          this.delivered[c]++;
          this.storedFood++;
          this.broodCredit[c]++;
          const td = a.tripDist[i];
          const eff = td > 1 ? Math.min(1, a.tripStraight[i] / td) : 1;
          this.effSum += eff;
          a.fitness[i] += 1 + eff;
          ang += Math.PI; // shared reflex: turn back toward the trail
          if (st === STATE_RETURNING) st = STATE_FORAGING;
        }
      } else if (!carrying && st !== STATE_NECROPHORESIS && st !== STATE_DEFENDING) {
        /* ---- crystals: pickup or depleted-site sealing ---- */
        const fc = this.foodCount;
        for (let f = 0; f < fc; f++) {
          if (food.active[f] === 0) continue;
          const dx = food.x[f] - nx;
          const dy = food.y[f] - ny;
          const fr = food.radius[f];
          if (dx * dx + dy * dy < fr * fr) {
            if (food.amount[f] >= 1) {
              const left = food.amount[f] - 1;
              food.amount[f] = left;
              carry[i] = 1;
              a.tripDist[i] = 0;
              a.tripTime[i] = 0;
              a.tripStraight[i] = Math.sqrt(nd2);
              a.targetX[i] = food.x[f];
              a.targetY[i] = food.y[f];
              this.pickups++;
              ang += Math.PI; // shared reflex: turn around with the load
              st = STATE_RETURNING;
              if (left < 1) {
                food.depletedTimer[f] = 18;
                food.radius[f] = 10;
              } else {
                food.radius[f] = 4 + 1.3 * Math.sqrt(left);
              }
            } else if (st !== STATE_TRAIL_DAMPING) {
              // depleted site: flip and seal the obsolete trail with noEntry
              st = STATE_TRAIL_DAMPING;
              stateTimer[i] = 1.2;
              ang += Math.PI;
            }
            break;
          }
        }
      }

      /* ---- stall / dead-end detection ---- */
      const stall = a.stallTimer[i] - dt;
      if (stall <= 0) {
        a.stallTimer[i] = 2.5;
        const mx = nx - a.anchorX[i];
        const my = ny - a.anchorY[i];
        if (mx * mx + my * my < 22 * 22 && nd2 > nestR2 * 4) {
          this.loops++;
          if (a.bump[i] > 25 && st <= STATE_RETURNING) {
            st = STATE_TRAIL_DAMPING;
            stateTimer[i] = 1.2;
            ang += Math.PI;
          }
        }
        a.anchorX[i] = nx;
        a.anchorY[i] = ny;
        a.bump[i] = 0;
      } else {
        a.stallTimer[i] = stall;
      }

      angle[i] = ang;
      state[i] = st;

      /* ---- gland secretion (every 4th tick, staggered) ---- */
      if (((tick + i) & 3) === 0) {
        const ci = ((ny * INV_CELL) | 0) * cols + ((nx * INV_CELL) | 0);
        if (wall[ci] === 0) {
          const k = 4 * dt;
          if (carry[i] !== 0 && e0 > 0.05) {
            const g = cur[0];
            const v = g[ci] + e0 * GLAND_RATE[0] * Math.exp(-a.tripTime[i] / FOOD_TAU) * k;
            g[ci] = v > 100 ? 100 : v;
            chActive[0] = 1;
          }
          if (e1 > 0.05) {
            const g = cur[1];
            const v = g[ci] + e1 * GLAND_RATE[1] * Math.exp(-a.timeSinceLastVisit[i] / HOME_TAU) * k;
            g[ci] = v > 100 ? 100 : v;
            chActive[1] = 1;
          }
          if (e2 > 0.05) {
            const g = cur[2];
            const v = g[ci] + e2 * GLAND_RATE[2] * k;
            g[ci] = v > 100 ? 100 : v;
            chActive[2] = 1;
          }
          if (e3 > 0.05) {
            const g = cur[3];
            const v = g[ci] + e3 * GLAND_RATE[3] * k;
            g[ci] = v > 100 ? 100 : v;
            chActive[3] = 1;
          }
          if (e4 > 0.05) {
            const g = cur[4];
            const v = g[ci] + e4 * GLAND_RATE[4] * k;
            g[ci] = v > 100 ? 100 : v;
            chActive[4] = 1;
          }
        }
      }
    }

    /* ---- reaction-diffusion ---- */
    if (this.halfRateDiffusion) {
      if ((tick & 1) === 0) grid.step(dt * 2);
    } else {
      grid.step(dt);
    }

    /* ---- brood, generations, founder bottleneck ---- */
    if (this.params.births) {
      for (let c = 0; c < this.nestCount; c++) {
        while (this.broodCredit[c] >= FOOD_PER_BIRTH) {
          if (this.alive >= this.params.maxPop || this.spawnAnt(c, true) < 0) {
            this.broodCredit[c] = FOOD_PER_BIRTH;
            break;
          }
          this.broodCredit[c] -= FOOD_PER_BIRTH;
          this.births++;
          if (this.births % BIRTHS_PER_GENERATION === 0) {
            this.colonyGeneration++;
            this.events |= EVENT_GENERATION;
          }
        }
      }
    }
    if (this.founderCooldown > 0) this.founderCooldown -= dt;
    if (this.params.founderWaves && this.alive < FOUNDER_THRESHOLD && this.founderCooldown <= 0 && this.time > 4) {
      const wave = Math.min(60, this.params.maxPop);
      for (let k = 0; k < wave; k++) this.spawnAnt(this.nestCount === 2 ? k & 1 : 0, true);
      this.births += wave;
      this.colonyGeneration++;
      this.founderBoost = FOUNDER_MUTATION_BOOST;
      this.founderCooldown = 30;
      this.events |= EVENT_FOUNDER | EVENT_GENERATION;
    }
  }

  /**
   * Colony fitness: food delivered + shortest-path efficiency (StraightDist / ActualDist) + homing progress
   * + trophallaxis / defence bonuses - wall scraping - looping stagnation - starvation.
   */
  colonyFitness(): number {
    const d = this.delivered[0] + this.delivered[1];
    const ticks = this.antTicks > 0 ? this.antTicks : 1;
    return (
      10 * d +
      4 * this.effSum +
      2 * this.pickups +
      3 * (this.homing / 150) +
      0.5 * Math.min(this.troph, 20) +
      2 * Math.min(this.defenseHits, 10) -
      60 * (this.wallHits / ticks) -
      0.15 * this.loops -
      2 * this.starved
    );
  }
}

export const CASTE_LABELS: readonly string[] = ['SCOUT', 'HAULER', 'WARDEN'];
export { CASTE_HAULER };
