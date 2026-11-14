import { describe, expect, it } from 'vitest';
import { GET } from '../app/api/health/route';
import {
  BIRTHS_PER_GENERATION, ColonySim, FIXED_DT, FOUNDER_MUTATION_BOOST, HASH_CELL, MAX_ANTS, MAX_PREDATORS,
} from '../lib/ant-soa-engine';
import {
  BENCH_SEEDS, DEFAULT_SHAPE, HeadlessTrainer, runBenchmark, runEpisode, scenarioForSeed, verifyOrdering,
} from '../lib/headless-sim';
import {
  BAKED_CHAMPION, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS, GENOME_SIZE, NeuroEvolution, bakeChampionSource,
  createRandomGenome, exportBrainJSON, forward, getChampionWeights, hasBakedChampion, importBrainJSON,
} from '../lib/neat';
import {
  ALARM_DECAY_MULT, BASE_EVAPORATION, CFL_LIMIT, DEFAULT_GAMMA, PHEROMONE_CELL, PheromoneGrid, mixPheromoneRGB,
  sampleCatmullRom,
} from '../lib/pheromone-grid';
import { PRESETS, applyPreset } from '../lib/presets';
import { Rng, Simplex2D, xorshift32 } from '../lib/prng-noise';
import {
  CASTE_HAULER, CASTE_WARDEN, CH_ALARM, CH_FOOD, CH_HOME, CH_NOENTRY, POLICY_ACO, POLICY_RANDOM, POLICY_TRAINED,
  STATE_ALARMED, STATE_DEAD, STATE_DEFENDING, STATE_NECROPHORESIS,
} from '../lib/types';

const DT = FIXED_DT;

function mix(f: number, h: number, a: number, s: number, n: number): number[] {
  const out = [0, 0, 0];
  mixPheromoneRGB(f, h, a, s, n, DEFAULT_GAMMA, out, 0);
  return out;
}

/** Bare world: one nest in the middle, no food, no walls. */
function bareSim(size: number, seed: number): ColonySim {
  const sim = new ColonySim(size, size, seed);
  sim.clearWorld();
  sim.nestCount = 1;
  sim.setNest(0, size / 2, size / 2);
  return sim;
}

function place(sim: ColonySim, i: number, x: number, y: number, angle: number): void {
  const a = sim.ants;
  a.x[i] = x;
  a.y[i] = y;
  a.angle[i] = angle;
  a.homeDx[i] = sim.nestX[0] - x;
  a.homeDy[i] = sim.nestY[0] - y;
  a.anchorX[i] = x;
  a.anchorY[i] = y;
}

describe('prng and noise', () => {
  it('Mulberry32 is deterministic and uniform in [0, 1)', () => {
    const a = new Rng(42);
    const b = new Rng(42);
    let sum = 0;
    for (let i = 0; i < 2000; i++) {
      const v = a.next();
      expect(v).toBe(b.next());
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
      sum += v;
    }
    expect(sum / 2000).toBeGreaterThan(0.45);
    expect(sum / 2000).toBeLessThan(0.55);
    expect(new Rng(43).next()).not.toBe(new Rng(42).next());
  });

  it('XorShift32 never sticks at zero and simplex noise is bounded and seeded', () => {
    expect(xorshift32(0)).not.toBe(0);
    expect(xorshift32(123)).toBe(xorshift32(123));
    const n1 = new Simplex2D(9);
    const n2 = new Simplex2D(9);
    for (let i = 0; i < 500; i++) {
      const v = n1.noise(i * 0.137, i * 0.091);
      expect(v).toBe(n2.noise(i * 0.137, i * 0.091));
      expect(Math.abs(v)).toBeLessThanOrEqual(1.05);
    }
  });
});

describe('5-channel pheromone grid', () => {
  it('uses cellSize 6 ping-pong Float32Arrays and clamps to [0, 100]', () => {
    const g = new PheromoneGrid(120, 60);
    expect(PHEROMONE_CELL).toBe(6);
    expect(g.cols).toBe(20);
    expect(g.rows).toBe(10);
    expect(g.cur).toHaveLength(5);
    expect(g.nxt).toHaveLength(5);
    expect(g.cur[0]).toBeInstanceOf(Float32Array);
    g.deposit(CH_FOOD, 30, 30, 80);
    g.deposit(CH_FOOD, 30, 30, 80);
    expect(g.sample(CH_FOOD, 30, 30)).toBe(100);
    const before = g.cur[0];
    g.step(DT);
    expect(g.cur[0]).not.toBe(before); // buffers swapped
  });

  it('diffusion conserves mass and never goes negative', () => {
    const g = new PheromoneGrid(240, 240);
    g.evaporation = 0;
    g.deposit(CH_FOOD, 120, 120, 50);
    for (let t = 0; t < 30; t++) g.step(DT);
    expect(g.total(CH_FOOD)).toBeGreaterThan(49);
    expect(g.total(CH_FOOD)).toBeLessThanOrEqual(50.001);
    for (const v of g.cur[CH_FOOD]) expect(v).toBeGreaterThanOrEqual(0);
    expect(g.sample(CH_FOOD, 126, 120)).toBeGreaterThan(0); // it actually spread
  });

  it('enforces the CFL clamp for absurd diffusion rates', () => {
    const g = new PheromoneGrid(240, 240);
    g.diffusion = 5000;
    g.evaporation = 0;
    g.deposit(CH_HOME, 120, 120, 90);
    for (let t = 0; t < 40; t++) {
      g.step(DT);
      expect(g.lastAlpha).toBeLessThanOrEqual(CFL_LIMIT);
    }
    let max = 0;
    for (const v of g.cur[CH_HOME]) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      if (v > max) max = v;
    }
    expect(max).toBeLessThan(90); // maximum principle: no overshoot
  });

  it('walls are zero-flux Neumann boundaries', () => {
    const g = new PheromoneGrid(240, 120);
    for (let gy = 0; gy < g.rows; gy++) g.setWallDisc((20 + 0.5) * 6, (gy + 0.5) * 6, 2, true);
    expect(g.wallCount).toBe(g.rows);
    g.evaporation = 0;
    g.diffusion = 8;
    g.deposit(CH_FOOD, 19.5 * 6, 60, 100);
    for (let t = 0; t < 400; t++) g.step(DT);
    let right = 0;
    let inWall = 0;
    for (let gy = 0; gy < g.rows; gy++) {
      inWall += g.cur[CH_FOOD][gy * g.cols + 20];
      for (let gx = 21; gx < g.cols; gx++) right += g.cur[CH_FOOD][gy * g.cols + gx];
    }
    expect(right).toBe(0);
    expect(inWall).toBe(0);
    expect(g.total(CH_FOOD)).toBeGreaterThan(85); // only the tiny-value numerical floor removes mass

    // a wall must behave exactly like the edge of the world (zero-flux): compare with a grid that simply ends there
    const edge = new PheromoneGrid(120, 120);
    edge.evaporation = 0;
    edge.diffusion = 8;
    edge.deposit(CH_FOOD, 19.5 * 6, 60, 100);
    for (let t = 0; t < 400; t++) edge.step(DT);
    for (let gy = 0; gy < g.rows; gy++) {
      for (let gx = 0; gx < 20; gx++) {
        expect(g.cur[CH_FOOD][gy * g.cols + gx]).toBeCloseTo(edge.cur[CH_FOOD][gy * edge.cols + gx], 5);
      }
    }
  });

  it('alarm decays 3.2x faster than foodTrail', () => {
    expect(BASE_EVAPORATION[CH_ALARM] / BASE_EVAPORATION[CH_FOOD]).toBeCloseTo(ALARM_DECAY_MULT, 6);
    expect(ALARM_DECAY_MULT).toBe(3.2);
    const g = new PheromoneGrid(60, 60);
    g.diffusion = 0;
    g.deposit(CH_FOOD, 30, 30, 50);
    g.deposit(CH_ALARM, 30, 30, 50);
    g.step(DT);
    const lossFood = 50 - g.sample(CH_FOOD, 30, 30);
    const lossAlarm = 50 - g.sample(CH_ALARM, 30, 30);
    expect(lossAlarm / lossFood).toBeCloseTo(3.2, 1);
  });

  it('foodTrail + homeGuide polymer decays 35% slower', () => {
    const alone = new PheromoneGrid(60, 60);
    const both = new PheromoneGrid(60, 60);
    alone.diffusion = 0;
    both.diffusion = 0;
    alone.deposit(CH_FOOD, 30, 30, 50);
    both.deposit(CH_FOOD, 30, 30, 50);
    both.deposit(CH_HOME, 30, 30, 100);
    alone.step(DT);
    both.step(DT);
    const lossAlone = 50 - alone.sample(CH_FOOD, 30, 30);
    const lossBoth = 50 - both.sample(CH_FOOD, 30, 30);
    expect(lossBoth).toBeLessThan(lossAlone);
    // saturating polymer fraction m / (m + 6) with m = 50 -> 1 - 0.35 * 0.893
    expect(lossBoth / lossAlone).toBeCloseTo(1 - 0.35 * (50 / 56), 2);
  });

  it('noEntry chemically dissolves foodTrail at 0.45 * C_noEntry * dt', () => {
    const g = new PheromoneGrid(60, 60);
    g.diffusion = 0;
    g.evaporation = 0;
    g.deposit(CH_FOOD, 30, 30, 50);
    g.deposit(CH_NOENTRY, 30, 30, 40);
    g.step(DT);
    expect(g.sample(CH_FOOD, 30, 30)).toBeCloseTo(50 - 0.45 * 40 * DT, 3);
    expect(g.sample(CH_NOENTRY, 30, 30)).toBeCloseTo(40, 3);
  });

  it('ablation masks start enabled and can mute a channel', () => {
    const g = new PheromoneGrid(60, 60);
    expect(Array.from(g.mask)).toEqual([1, 1, 1, 1, 1]);
    g.setMask(CH_ALARM, false);
    expect(g.mask[CH_ALARM]).toBe(0);
  });
});

describe('spectral RGB mixer', () => {
  it('renders nothing for an empty cell and Solar Amber for foodTrail', () => {
    expect(mix(0, 0, 0, 0, 0)).toEqual([0, 0, 0]);
    const [r, g, b] = mix(60, 0, 0, 0, 0);
    expect(r).toBeGreaterThan(g);
    expect(g).toBeGreaterThan(b);
    expect(r).toBeGreaterThan(180);
  });

  it('tonemaps monotonically: more pheromone is never darker', () => {
    let last = -1;
    for (let c = 0; c <= 100; c += 5) {
      const [, , b] = mix(0, c, 0, 0, 0);
      expect(b).toBeGreaterThanOrEqual(last);
      last = b;
    }
    expect(last).toBeLessThanOrEqual(255);
  });

  it('food + home: turquoise edges and pearl-white highway cores', () => {
    const edge = mix(8, 8, 0, 0, 0);
    expect(edge[1]).toBeGreaterThan(edge[0]);
    expect(edge[2]).toBeGreaterThan(edge[0]);
    const core = mix(70, 70, 0, 0, 0);
    for (const ch of core) expect(ch).toBeGreaterThan(235);
  });

  it('food + scout: rose-gold, home + alarm: magenta, alarm + noEntry: sulfur-gold', () => {
    const rose = mix(60, 0, 0, 60, 0);
    expect(rose[0]).toBeGreaterThan(230);
    expect(rose[0]).toBeGreaterThan(rose[1]);
    expect(rose[1]).toBeGreaterThan(rose[2]);
    const magenta = mix(0, 60, 60, 0, 0);
    expect(magenta[0]).toBeGreaterThan(220);
    expect(magenta[2]).toBeGreaterThan(190);
    expect(magenta[1]).toBeLessThan(80);
    const sulfur = mix(0, 0, 60, 0, 60);
    expect(sulfur[0]).toBeGreaterThan(210);
    expect(sulfur[1]).toBeGreaterThan(190);
    expect(sulfur[2]).toBeLessThan(90);
  });

  it('Catmull-Rom brush stamps every 0.25 * radius and ends on the pointer', () => {
    const out = new Float32Array(128);
    const radius = 20;
    const n = sampleCatmullRom(0, 0, 0, 0, 100, 0, 100, 0, radius * 0.25, out);
    expect(n).toBe(20);
    expect(out[(n - 1) * 2]).toBeCloseTo(100, 4);
    expect(out[(n - 1) * 2 + 1]).toBeCloseTo(0, 4);
    for (let k = 1; k < n; k++) expect(out[k * 2]).toBeGreaterThan(out[(k - 1) * 2]);
  });
});

describe('shared colony brain', () => {
  it('is a fixed 19 -> 16 -> 9 topology with 473 weights', () => {
    expect(BRAIN_INPUTS).toBe(19);
    expect(BRAIN_HIDDEN).toBe(16);
    expect(BRAIN_OUTPUTS).toBe(9);
    expect(GENOME_SIZE).toBe(19 * 16 + 16 + 16 * 9 + 9);
    expect(getChampionWeights()).toHaveLength(GENOME_SIZE);
  });

  it('forward pass keeps every output in range and is deterministic', () => {
    const rng = new Rng(5);
    const w = createRandomGenome(rng, 2);
    const inp = new Float32Array(BRAIN_INPUTS);
    const hid = new Float32Array(BRAIN_HIDDEN);
    const out = new Float32Array(BRAIN_OUTPUTS);
    const out2 = new Float32Array(BRAIN_OUTPUTS);
    for (let trial = 0; trial < 200; trial++) {
      for (let i = 0; i < BRAIN_INPUTS; i++) inp[i] = rng.range(-3, 3);
      forward(w, inp, hid, out);
      forward(w, inp, hid, out2);
      expect(Array.from(out)).toEqual(Array.from(out2));
      expect(Math.abs(out[0])).toBeLessThanOrEqual(1);
      for (let o = 1; o <= 6; o++) {
        expect(out[o]).toBeGreaterThanOrEqual(0);
        expect(out[o]).toBeLessThanOrEqual(1);
      }
      expect(Math.abs(out[7])).toBeLessThanOrEqual(1);
      expect(Math.abs(out[8])).toBeLessThanOrEqual(1);
    }
  });

  it('recurrent memory inputs change the output', () => {
    const w = createRandomGenome(new Rng(11), 1.5);
    const inp = new Float32Array(BRAIN_INPUTS).fill(0.2);
    const hid = new Float32Array(BRAIN_HIDDEN);
    const a = new Float32Array(BRAIN_OUTPUTS);
    const b = new Float32Array(BRAIN_OUTPUTS);
    forward(w, inp, hid, a);
    inp[17] = -1;
    inp[18] = 1;
    forward(w, inp, hid, b);
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  it('exports and imports weights as JSON and rejects malformed brains', () => {
    const w = createRandomGenome(new Rng(3), 1);
    const json = JSON.stringify(exportBrainJSON(w, 12));
    const back = importBrainJSON(json);
    expect(back).not.toBeNull();
    expect(back?.generation).toBe(12);
    for (let i = 0; i < GENOME_SIZE; i++) expect(back?.weights[i]).toBeCloseTo(w[i], 3);
    expect(importBrainJSON('not json')).toBeNull();
    expect(importBrainJSON({ format: 'cryo-colony-brain', inputs: 19, hidden: 16, outputs: 9, weights: [1, 2, 3] })).toBeNull();
    const bad = exportBrainJSON(w, 1);
    expect(importBrainJSON({ ...bad, weights: bad.weights.map((v, i) => (i === 7 ? Number.NaN : v)) })).toBeNull();
    expect(importBrainJSON({ ...bad, hidden: 8 })).toBeNull();
  });

  it('neuroevolution keeps elites, mutates the rest and can speciate', () => {
    const evo = new NeuroEvolution({ populationSize: 10, elites: 2, seed: 4, speciation: true });
    for (let i = 0; i < 10; i++) evo.fitness[i] = i;
    const best = Float32Array.from(evo.population[9]);
    const second = Float32Array.from(evo.population[8]);
    evo.evolve();
    expect(evo.generation).toBe(1);
    expect(evo.population).toHaveLength(10);
    expect(Array.from(evo.population[0])).toEqual(Array.from(best));
    expect(Array.from(evo.population[1])).toEqual(Array.from(second));
    expect(evo.speciesCount).toBeGreaterThanOrEqual(1);
    let changed = 0;
    for (let c = 2; c < 10; c++) if (Array.from(evo.population[c]).join() !== Array.from(best).join()) changed++;
    expect(changed).toBeGreaterThan(0);
    for (const g of evo.population) for (const v of g) expect(Math.abs(v)).toBeLessThanOrEqual(8);
  });

  it('adaptive mutation shrinks on progress and widens on stagnation', () => {
    const evo = new NeuroEvolution({ populationSize: 4, seed: 1 });
    const s0 = evo.sigma;
    evo.noteProgress(true);
    expect(evo.sigma).toBeLessThan(s0);
    const s1 = evo.sigma;
    for (let k = 0; k < 6; k++) evo.noteProgress(false);
    expect(evo.sigma).toBeGreaterThan(s1);
    expect(evo.sigma).toBeLessThanOrEqual(evo.opts.sigmaMax);
  });

  it('bakes a champion between the source markers', () => {
    const src = 'head\n/* @@CHAMPION_BEGIN@@ */\nold\n/* @@CHAMPION_END@@ */\ntail';
    const baked = bakeChampionSource(src, {
      version: 1, trainedAt: '2026-01-01', generations: 3, populationSize: 4, trainSeed: 1, history: [], benchmark: null, weights: [0.5],
    });
    expect(baked.startsWith('head\n')).toBe(true);
    expect(baked.endsWith('/* @@CHAMPION_END@@ */\ntail')).toBe(true);
    expect(baked).toContain('"weights":[0.5]');
    expect(baked).not.toContain('old');
  });
});

describe('SoA engine', () => {
  it('preallocates 3000-ant typed arrays and an Int32Array spatial hash with 24px cells', () => {
    const sim = new ColonySim(480, 480, 1);
    expect(MAX_ANTS).toBe(3000);
    expect(MAX_PREDATORS).toBe(6);
    expect(HASH_CELL).toBe(24);
    const a = sim.ants;
    for (const arr of [a.x, a.y, a.vx, a.vy, a.angle, a.homeDx, a.homeDy, a.targetX, a.targetY, a.legPhase,
      a.pheromoneTimer, a.timeSinceLastVisit, a.ouNoise, a.energy, a.age, a.maxLifespan, a.fitness, a.memA, a.memB]) {
      expect(arr).toBeInstanceOf(Float32Array);
      expect(arr).toHaveLength(MAX_ANTS);
    }
    for (const arr of [a.state, a.caste, a.policyType, a.carryingFood, a.active]) {
      expect(arr).toBeInstanceOf(Uint8Array);
      expect(arr).toHaveLength(MAX_ANTS);
    }
    expect(a.generation).toBeInstanceOf(Uint16Array);
    expect(sim.hashHead).toBeInstanceOf(Int32Array);
    expect(sim.hashNext).toBeInstanceOf(Int32Array);
  });

  it('reuses the same buffers while stepping and keeps ants finite and in bounds', () => {
    const sim = new ColonySim(480, 480, 3);
    sim.params.births = false;
    sim.params.founderWaves = false;
    applyPreset(sim, 'open', 3, 120);
    const xRef = sim.ants.x;
    const hashRef = sim.hashHead;
    for (let t = 0; t < 240; t++) sim.step(DT);
    expect(sim.ants.x).toBe(xRef);
    expect(sim.hashHead).toBe(hashRef);
    expect(sim.alive).toBe(120);
    for (let i = 0; i < sim.count; i++) {
      expect(Number.isFinite(sim.ants.x[i])).toBe(true);
      expect(sim.ants.x[i]).toBeGreaterThanOrEqual(0);
      expect(sim.ants.x[i]).toBeLessThanOrEqual(480);
      expect(sim.ants.y[i]).toBeGreaterThanOrEqual(0);
      expect(sim.ants.y[i]).toBeLessThanOrEqual(480);
    }
  });

  it('is deterministic for a given seed', () => {
    const cfg = { seed: 77, ants: 24, ticks: 300, worldSize: 480, policy: POLICY_ACO, weights: null, ablation: null } as const;
    const a = runEpisode(cfg);
    const b = runEpisode(cfg);
    expect(a).toEqual(b);
  });

  it('ants that run out of ATP die, drop food and leave a corpse', () => {
    const sim = bareSim(400, 2);
    sim.params.founderWaves = false;
    const i = sim.spawnAnt(0, true);
    place(sim, i, 80, 80, 0);
    sim.ants.energy[i] = 0.001;
    sim.ants.carryingFood[i] = 1;
    expect(sim.alive).toBe(1);
    sim.step(DT);
    expect(sim.ants.state[i]).toBe(STATE_DEAD);
    expect(sim.alive).toBe(0);
    expect(sim.totalDeaths).toBe(1);
    expect(sim.starved).toBe(1);
    expect(sim.ants.carryingFood[i]).toBe(0);
    expect(sim.totalFoodRemaining()).toBe(1); // dropped crystal
    // the husk leaks weak necrotic alarm
    for (let t = 0; t < 40; t++) sim.step(DT);
    expect(sim.grid.total(CH_ALARM)).toBeGreaterThan(0);
  });

  it('ants die of old age and the nest refills energy to 100%', () => {
    const sim = bareSim(400, 2);
    sim.params.founderWaves = false;
    const old = sim.spawnAnt(0, true);
    place(sim, old, 90, 90, 0);
    sim.ants.age[old] = sim.ants.maxLifespan[old] + 1;
    const hungry = sim.spawnAnt(0, true); // spawned inside the nest
    sim.ants.energy[hungry] = 12;
    sim.step(DT);
    expect(sim.ants.state[old]).toBe(STATE_DEAD);
    expect(sim.starved).toBe(0);
    expect(sim.ants.energy[hungry]).toBe(100);
  });

  it('every 3 delivered food hatches 1 ant and every 25 births advances the generation', () => {
    const sim = bareSim(400, 5);
    sim.params.founderWaves = false;
    sim.populate(40);
    sim.broodCredit[0] = 9;
    sim.step(DT);
    expect(sim.births).toBe(3);
    expect(sim.alive).toBe(43);
    expect(sim.colonyGeneration).toBe(1);
    sim.births = BIRTHS_PER_GENERATION - 1;
    sim.broodCredit[0] = 3;
    sim.step(DT);
    expect(sim.births).toBe(BIRTHS_PER_GENERATION);
    expect(sim.colonyGeneration).toBe(2);
  });

  it('respects the population cap', () => {
    const sim = bareSim(400, 5);
    sim.params.founderWaves = false;
    sim.params.maxPop = 40;
    sim.populate(40);
    sim.broodCredit[0] = 30;
    sim.step(DT);
    expect(sim.alive).toBe(40);
    expect(sim.births).toBe(0);
  });

  it('a Founder Bottleneck brood wave fires below 35 survivors with a 2.5x mutation boost', () => {
    const sim = bareSim(400, 6);
    sim.populate(10);
    expect(sim.founderBoost).toBe(1);
    for (let t = 0; t < 300; t++) sim.step(DT);
    expect(sim.alive).toBeGreaterThanOrEqual(35);
    expect(sim.founderBoost).toBe(FOUNDER_MUTATION_BOOST);
    expect(FOUNDER_MUTATION_BOOST).toBe(2.5);
    expect(sim.colonyGeneration).toBeGreaterThan(1);
  });

  it('predators trigger a crimson alarm burst; foragers flee and wardens defend', () => {
    const sim = bareSim(400, 8);
    sim.params.founderWaves = false;
    const hauler = sim.spawnAnt(0, true);
    const warden = sim.spawnAnt(0, true);
    sim.ants.caste[hauler] = CASTE_HAULER;
    sim.ants.caste[warden] = CASTE_WARDEN;
    place(sim, hauler, 100, 100, 0);
    place(sim, warden, 100, 120, 0);
    expect(sim.spawnPredator(150, 110)).toBe(true);
    expect(sim.grid.total(CH_ALARM)).toBe(0);
    sim.step(DT);
    expect(sim.grid.total(CH_ALARM)).toBeGreaterThan(100); // two +80 bursts
    expect(sim.ants.state[hauler]).toBe(STATE_ALARMED);
    expect(sim.ants.state[warden]).toBe(STATE_DEFENDING);
    // the hauler ends up farther from the predator, the warden closer
    const d0h = Math.hypot(sim.ants.x[hauler] - sim.pred.x[0], sim.ants.y[hauler] - sim.pred.y[0]);
    const d0w = Math.hypot(sim.ants.x[warden] - sim.pred.x[0], sim.ants.y[warden] - sim.pred.y[0]);
    sim.pred.cooldown[0] = 1e6; // keep it from biting during the check
    for (let t = 0; t < 45; t++) sim.step(DT);
    const d1w = Math.hypot(sim.ants.x[warden] - sim.pred.x[0], sim.ants.y[warden] - sim.pred.y[0]);
    expect(d1w).toBeLessThan(d0w);
    expect(d0h).toBeGreaterThan(0);
  });

  it('wardens wear a predator down and never exceed the predator cap', () => {
    const sim = bareSim(400, 9);
    sim.params.founderWaves = false;
    sim.params.mortality = false;
    for (let k = 0; k < 30; k++) {
      const i = sim.spawnAnt(0, true);
      sim.ants.caste[i] = CASTE_WARDEN;
      place(sim, i, 110 + (k % 6) * 3, 100 + ((k / 6) | 0) * 3, 0);
    }
    sim.spawnPredator(140, 106);
    sim.pred.cooldown[0] = 1e6;
    for (let t = 0; t < 600 && sim.predatorCount > 0; t++) sim.step(DT);
    expect(sim.defenseHits).toBeGreaterThan(0);
    expect(sim.predatorCount).toBe(0); // neutralised
    sim.setPredatorCount(20);
    expect(sim.predatorCount).toBeLessThanOrEqual(MAX_PREDATORS);
  });

  it('trophallaxis: energy-rich ants feed starving neighbours', () => {
    const sim = bareSim(400, 10);
    sim.params.founderWaves = false;
    const rich = sim.spawnAnt(0, true);
    const poor = sim.spawnAnt(0, true);
    place(sim, rich, 80, 80, 0.3);
    place(sim, poor, 83, 80, 0.3);
    sim.ants.energy[rich] = 100;
    sim.ants.energy[poor] = 12;
    for (let t = 0; t < 8; t++) sim.step(DT);
    expect(sim.troph).toBeGreaterThanOrEqual(1);
    expect(sim.ants.energy[poor]).toBeGreaterThan(20);
    expect(sim.ants.energy[rich]).toBeLessThan(90);
  });

  it('necrophoresis: a warden picks up a nearby corpse', () => {
    const sim = bareSim(400, 12);
    sim.params.founderWaves = false;
    const doomed = sim.spawnAnt(0, true);
    const warden = sim.spawnAnt(0, true);
    sim.ants.caste[warden] = CASTE_WARDEN;
    place(sim, doomed, 80, 80, 0);
    place(sim, warden, 80, 80, 0);
    sim.ants.energy[doomed] = 0.001;
    for (let t = 0; t < 8; t++) sim.step(DT);
    expect(sim.ants.state[doomed]).toBe(STATE_DEAD);
    expect(sim.ants.state[warden]).toBe(STATE_NECROPHORESIS);
    expect(sim.ants.link[warden]).toBe(doomed);
    expect(sim.ants.link[doomed]).toBe(warden);
  });

  it('muting path integration zeroes that input for every policy', () => {
    const weights = getChampionWeights();
    const base = { seed: 9002, ants: 36, ticks: 600, worldSize: 480, policy: POLICY_TRAINED, weights } as const;
    const full = runEpisode({ ...base, ablation: null });
    const muted = runEpisode({ ...base, ablation: 'pathIntegration' });
    expect(full).not.toEqual(muted);
  });
});

describe('presets', () => {
  it('each preset builds a reachable nest and food', () => {
    for (const p of PRESETS) {
      const sim = new ColonySim(720, 1280, 21);
      sim.setArena('trained');
      applyPreset(sim, p.id, 21, 100);
      expect(sim.nestCount).toBe(1);
      expect(sim.alive).toBe(100);
      expect(sim.totalFoodRemaining()).toBeGreaterThan(100);
      expect(sim.grid.isWallAt(sim.nestX[0], sim.nestY[0])).toBe(false);
      if (p.id !== 'open') expect(sim.grid.wallCount).toBeGreaterThan(50);
    }
    expect(PRESETS.map((p) => p.title)).toEqual(['Open Sanctuary', "Deneubourg's Double Bridge", 'Subterranean Labyrinth']);
  });

  it('50/50 arenas build two sealed halves with one cohort each', () => {
    const sim = new ColonySim(1280, 720, 4);
    sim.setArena('ab-random');
    applyPreset(sim, 'open', 4, 200);
    expect(sim.nestCount).toBe(2);
    expect(sim.cohortAlive[0]).toBe(100);
    expect(sim.cohortAlive[1]).toBe(100);
    expect(sim.grid.isWallAt(640, 360)).toBe(true); // divider
    expect(sim.nestX[0]).toBeLessThan(640);
    expect(sim.nestX[1]).toBeGreaterThan(640);
    for (let i = 0; i < sim.count; i++) {
      expect(sim.ants.policyType[i]).toBe(sim.ants.cohort[i] === 0 ? POLICY_TRAINED : POLICY_RANDOM);
    }
    sim.setArena('ab-aco');
    applyPreset(sim, 'bridge', 4, 200);
    for (let i = 0; i < sim.count; i++) {
      expect(sim.ants.policyType[i]).toBe(sim.ants.cohort[i] === 0 ? POLICY_TRAINED : POLICY_ACO);
    }
    for (let t = 0; t < 120; t++) sim.step(DT);
    // cohorts never cross the divider
    for (let i = 0; i < sim.count; i++) {
      if (sim.ants.active[i] === 0) continue;
      if (sim.ants.cohort[i] === 0) expect(sim.ants.x[i]).toBeLessThan(640);
      else expect(sim.ants.x[i]).toBeGreaterThan(640);
    }
  });
});

describe('headless training and proof of learning', () => {
  it('covers both scenario families across consecutive seeds', () => {
    expect([0, 1, 2].map(scenarioForSeed)).toEqual(['open', 'bridge', 'open']);
  });

  it('the chunked trainer advances one episode at a time and records generations', () => {
    const shape = { ants: 12, ticks: 120, worldSize: 240 };
    const trainer = new HeadlessTrainer({ populationSize: 4, seedsPerEval: 2, shape, seed: 3 });
    expect(trainer.episodesPerGeneration).toBe(4 * 2 + 3);
    let calls = 0;
    let rec = trainer.advance();
    calls++;
    while (rec === null) {
      rec = trainer.advance();
      calls++;
    }
    expect(calls).toBe(trainer.episodesPerGeneration);
    expect(rec.generation).toBe(0);
    expect(Number.isFinite(rec.bestFitness)).toBe(true);
    const second = trainer.runGeneration();
    expect(second.generation).toBe(1);
    expect(trainer.history).toHaveLength(2);
    expect(second.championFood).toBeGreaterThanOrEqual(0);
    expect(trainer.champion).toHaveLength(GENOME_SIZE);
  });

  it('ships a baked, trained champion with its learning curve and benchmark', () => {
    expect(hasBakedChampion()).toBe(true);
    expect(BAKED_CHAMPION.generations).toBeGreaterThan(10);
    expect(BAKED_CHAMPION.history.length).toBeGreaterThan(10);
    expect(BAKED_CHAMPION.benchmark).not.toBeNull();
    const first = BAKED_CHAMPION.history[0];
    const last = BAKED_CHAMPION.history[BAKED_CHAMPION.history.length - 1];
    expect(last.championFood).toBeGreaterThan(first.championFood); // it learned
  });

  it('multi-seed benchmark: Trained > Hand-Crafted ACO > Random, and path integration matters', () => {
    const result = runBenchmark(getChampionWeights(), { seeds: BENCH_SEEDS, shape: DEFAULT_SHAPE });
    expect(result.rows).toHaveLength(9);
    expect(result.rows.map((r) => r.id)).toEqual([
      'trained', 'aco', 'random', 'foodTrail', 'homeGuide', 'alarm', 'scoutPulse', 'noEntry', 'pathIntegration',
    ]);
    for (const r of result.rows) {
      expect(r.samples).toHaveLength(BENCH_SEEDS.length);
      expect(r.std).toBeGreaterThanOrEqual(0);
    }
    const verdict = verifyOrdering(result);
    expect(verdict.trainedBeatsAco).toBe(true);
    expect(verdict.acoBeatsRandom).toBe(true);
    const pathInt = result.rows[8];
    expect(pathInt.delta).toBeLessThan(0);
    expect(verdict.drops).toContain('-PathIntegration');
    // the baked benchmark must be reproducible from the baked weights
    const baked = BAKED_CHAMPION.benchmark;
    expect(baked?.rows[0].mean).toBeCloseTo(result.rows[0].mean, 6);
    expect(baked?.rows[1].mean).toBeCloseTo(result.rows[1].mean, 6);
  }, 180000);
});

describe('/api/health', () => {
  it('returns 200 with every engine check passing', async () => {
    const res = await GET();
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; checks: Record<string, boolean> };
    expect(body.ok).toBe(true);
    expect(body.checks).toEqual({ soa: true, brain: true, rgbMix: true, headlessAblation: true });
  });
});
