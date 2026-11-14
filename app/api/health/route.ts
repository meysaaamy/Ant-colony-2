import { ColonySim, FIXED_DT, MAX_ANTS } from '../../../lib/ant-soa-engine';
import { runEpisode } from '../../../lib/headless-sim';
import { BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS, GENOME_SIZE, forward, getChampionWeights, hasBakedChampion } from '../../../lib/neat';
import { DEFAULT_GAMMA, mixPheromoneRGB } from '../../../lib/pheromone-grid';
import { applyPreset } from '../../../lib/presets';
import { POLICY_TRAINED } from '../../../lib/types';

export const dynamic = 'force-dynamic';

/**
 * Liveness + engine self-check:
 *  1. SoA engine boots, steps and keeps finite state
 *  2. lib/neat.ts forward pass produces in-range outputs
 *  3. 5-channel RGB mixer yields amber for foodTrail and a pearl core for foodTrail + homeGuide
 *  4. a tiny headless ablation pair (full brain vs path integration muted) completes
 */
export async function GET() {
  try {
    // 1: SoA
    const sim = new ColonySim(240, 240, 7);
    applyPreset(sim, 'open', 7, 32);
    for (let t = 0; t < 90; t++) sim.step(FIXED_DT);
    let finite = sim.ants.x.length === MAX_ANTS && sim.alive > 0;
    for (let i = 0; i < sim.count; i++) {
      if (!Number.isFinite(sim.ants.x[i]) || !Number.isFinite(sim.ants.y[i]) || !Number.isFinite(sim.ants.angle[i])) finite = false;
    }

    // 2: brain forward pass
    const weights = getChampionWeights();
    const inp = new Float32Array(BRAIN_INPUTS).fill(0.25);
    const hid = new Float32Array(BRAIN_HIDDEN);
    const out = new Float32Array(BRAIN_OUTPUTS);
    forward(weights, inp, hid, out);
    let brainOk = weights.length === GENOME_SIZE && Math.abs(out[0]) <= 1;
    for (let o = 1; o <= 6; o++) if (!(out[o] >= 0 && out[o] <= 1)) brainOk = false;
    for (let o = 0; o < BRAIN_OUTPUTS; o++) if (!Number.isFinite(out[o])) brainOk = false;

    // 3: RGB mix
    const rgb = [0, 0, 0, 0, 0, 0];
    mixPheromoneRGB(60, 0, 0, 0, 0, DEFAULT_GAMMA, rgb, 0);
    mixPheromoneRGB(70, 70, 0, 0, 0, DEFAULT_GAMMA, rgb, 3);
    const mixOk = rgb[0] > rgb[1] && rgb[1] > rgb[2] && rgb[3] > 200 && rgb[4] > 200 && rgb[5] > 200;

    // 4: headless ablation
    const base = { seed: 3, ants: 16, ticks: 240, worldSize: 240, policy: POLICY_TRAINED, weights } as const;
    const full = runEpisode({ ...base, ablation: null });
    const ablated = runEpisode({ ...base, ablation: 'pathIntegration' });
    const headlessOk = Number.isFinite(full.fitness) && Number.isFinite(ablated.fitness);

    const ok = finite && brainOk && mixOk && headlessOk;
    return Response.json(
      {
        ok,
        checks: { soa: finite, brain: brainOk, rgbMix: mixOk, headlessAblation: headlessOk },
        champion: { baked: hasBakedChampion(), genome: GENOME_SIZE },
        sample: { alive: sim.alive, foodFull: full.food, foodNoPathIntegration: ablated.food },
      },
      { status: ok ? 200 : 500 },
    );
  } catch (err) {
    return Response.json({ ok: false, error: err instanceof Error ? err.message : 'unknown error' }, { status: 500 });
  }
}
