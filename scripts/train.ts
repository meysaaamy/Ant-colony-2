/**
 * CLI offline trainer and multi-seed benchmark runner.
 *
 *   npx tsx scripts/train.ts                       evolve, benchmark, bake champion into lib/neat.ts
 *   npx tsx scripts/train.ts --gens 60 --pop 24    tune the run
 *   npx tsx scripts/train.ts --budget 300          stop evolving after N seconds
 *   npx tsx scripts/train.ts --out a.json --no-bake   write a checkpoint instead of baking
 *   npx tsx scripts/train.ts --from a.json,b.json  pick the best island on validation seeds, benchmark, bake
 *   npx tsx scripts/train.ts --bench-only          benchmark whatever is currently baked
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  BENCH_SEEDS, CURVE_SEEDS, DEFAULT_SHAPE, HeadlessTrainer, evaluateBaseline, evaluateWeights, runBenchmark, verifyOrdering,
} from '../lib/headless-sim';
import { GENOME_SIZE, bakeChampionSource, getChampionWeights, hasBakedChampion, BAKED_CHAMPION } from '../lib/neat';
import { POLICY_ACO, POLICY_RANDOM } from '../lib/types';
import type { BakedChampion, BenchmarkResult, GenerationRecord } from '../lib/types';

interface Checkpoint {
  seed: number;
  populationSize: number;
  generations: number;
  history: GenerationRecord[];
  weights: number[];
}

/** Validation seeds for choosing between islands: disjoint from both curve and benchmark seeds. */
const VALIDATION_SEEDS: readonly number[] = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010, 7011, 7012];

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

function flag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

function round4(w: Float32Array): number[] {
  const out: number[] = [];
  for (let i = 0; i < w.length; i++) out.push(Math.round(w[i] * 10000) / 10000);
  return out;
}

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + ' '.repeat(n - s.length);
}

function printBenchmark(b: BenchmarkResult): void {
  console.log(`\nHeadless benchmark: ${b.seeds.length} held-out seeds [${b.seeds.join(', ')}], ${b.ants} ants x ${b.ticks} ticks`);
  console.log(`${pad('condition', 22)}${pad('food mean', 12)}${pad('std', 9)}${pad('delta', 10)}samples`);
  for (const r of b.rows) {
    console.log(
      `${pad(r.label, 22)}${pad(r.mean.toFixed(1), 12)}${pad(r.std.toFixed(1), 9)}${pad((r.delta >= 0 ? '+' : '') + r.delta.toFixed(1), 10)}${r.samples.join(' ')}`,
    );
  }
  const v = verifyOrdering(b);
  console.log(`\nVERIFY Trained > ACO   : ${v.trainedBeatsAco ? 'PASS' : 'FAIL'}  (${v.trained.toFixed(1)} vs ${v.aco.toFixed(1)})`);
  console.log(`VERIFY ACO > Random    : ${v.acoBeatsRandom ? 'PASS' : 'FAIL'}  (${v.aco.toFixed(1)} vs ${v.random.toFixed(1)})`);
  console.log(`VERIFY ablation drops  : ${v.drops.length > 0 ? v.drops.join(', ') : 'none'}`);
}

function bake(champion: BakedChampion): void {
  const file = resolve(process.cwd(), 'lib/neat.ts');
  const src = readFileSync(file, 'utf8');
  writeFileSync(file, bakeChampionSource(src, champion), 'utf8');
  console.log(`\nBaked champion (${champion.generations} generations) into lib/neat.ts`);
}

function main(): void {
  const gens = parseInt(arg('gens', '40'), 10);
  const pop = parseInt(arg('pop', '24'), 10);
  const seed = parseInt(arg('seed', '1'), 10);
  const budget = parseFloat(arg('budget', '100000'));
  const out = arg('out', '');
  const from = arg('from', '');
  const doBake = !flag('no-bake');
  const shape = DEFAULT_SHAPE;

  let weights: Float32Array;
  let history: GenerationRecord[] = [];
  let generations = 0;
  let trainSeed = seed;
  let populationSize = pop;

  if (flag('bench-only')) {
    if (!hasBakedChampion()) console.log('No baked champion yet: benchmarking the untrained fallback genome.');
    weights = getChampionWeights();
    history = BAKED_CHAMPION.history;
    generations = BAKED_CHAMPION.generations;
    printBenchmark(runBenchmark(weights, { seeds: BENCH_SEEDS, shape }));
    return;
  }

  if (from !== '') {
    let best: Checkpoint | null = null;
    let bestScore = -Infinity;
    for (const f of from.split(',')) {
      const cp = JSON.parse(readFileSync(resolve(process.cwd(), f), 'utf8')) as Checkpoint;
      if (cp.weights.length !== GENOME_SIZE) continue;
      const s = evaluateWeights(Float32Array.from(cp.weights), VALIDATION_SEEDS, shape);
      console.log(`island ${f}: seed ${cp.seed}, ${cp.generations} gens, validation food ${s.food.toFixed(1)}, fitness ${s.fitness.toFixed(1)}`);
      if (s.fitness > bestScore) {
        bestScore = s.fitness;
        best = cp;
      }
    }
    if (!best) throw new Error('no usable checkpoint');
    weights = Float32Array.from(best.weights);
    history = best.history;
    generations = best.generations;
    trainSeed = best.seed;
    populationSize = best.populationSize;
    console.log(`selected island seed ${best.seed}`);
  } else {
    const warm = flag('warm') && hasBakedChampion() ? getChampionWeights() : null;
    const trainer = new HeadlessTrainer({
      populationSize: pop, seed, shape, speciation: flag('speciate'), warmStart: warm,
      startGeneration: warm ? BAKED_CHAMPION.generations : 0,
    });
    if (warm) history = [...BAKED_CHAMPION.history];
    const t0 = Date.now();
    trainer.computeBaselines();
    console.log(
      `train seed ${seed} | pop ${pop} | ${shape.ants} ants x ${shape.ticks} ticks | curve seeds [${CURVE_SEEDS.join(', ')}]` +
      ` | ACO ${trainer.acoFood.toFixed(1)} | Random ${trainer.randomFood.toFixed(1)}`,
    );
    for (let g = 0; g < gens; g++) {
      const rec = trainer.runGeneration();
      const secs = (Date.now() - t0) / 1000;
      console.log(
        `gen ${pad(String(rec.generation), 4)}| fit best ${pad(rec.bestFitness.toFixed(1), 8)} mean ${pad(rec.meanFitness.toFixed(1), 8)}` +
        `| food ${pad(rec.bestFood.toFixed(1), 6)} champ ${pad(rec.championFood.toFixed(1), 6)}` +
        `| ACO ${rec.acoFood.toFixed(1)} RND ${rec.randomFood.toFixed(1)} | sigma ${rec.sigma.toFixed(3)} | ${secs.toFixed(0)}s`,
      );
      if (out !== '') {
        const cp: Checkpoint = {
          seed, populationSize: pop, generations: trainer.generation,
          history: [...history, ...trainer.history], weights: round4(trainer.champion),
        };
        writeFileSync(resolve(process.cwd(), out), JSON.stringify(cp), 'utf8');
      }
      if (secs > budget) {
        console.log(`time budget of ${budget}s reached after ${g + 1} generations`);
        break;
      }
    }
    weights = trainer.champion;
    history = [...history, ...trainer.history];
    generations = trainer.generation;
  }

  // benchmark exactly what gets baked (4-decimal rounded weights)
  const rounded = round4(weights);
  const finalWeights = Float32Array.from(rounded);
  const benchmark = runBenchmark(finalWeights, { seeds: BENCH_SEEDS, shape });
  printBenchmark(benchmark);
  const acoCurve = evaluateBaseline(POLICY_ACO, CURVE_SEEDS, shape);
  const rndCurve = evaluateBaseline(POLICY_RANDOM, CURVE_SEEDS, shape);
  console.log(`curve-seed baselines: ACO ${acoCurve.toFixed(1)}, Random ${rndCurve.toFixed(1)}`);

  if (doBake) {
    bake({
      version: 1,
      trainedAt: new Date().toISOString().slice(0, 10),
      generations,
      populationSize,
      trainSeed,
      history,
      benchmark,
      weights: rounded,
    });
  }
}

main();
