/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2.zip/components/TelemetryPill.tsx)
 * Why best: رابط الفابلهی séd interacting با sim: ۶ ابزار، صفحه ۳냅، HUD،_top pills
 * Borrow from other zips: mobile: BioSlider (دابل‌تاپ reset)؛ astramax: D3 dispersion graph، تجربیات /api/health
 * Notes: براي ادجام به best-of/components/TelemetryPill.tsx
 * Status: verbatim copy — adapt import paths when merging into app/
 */

'use client';

import type { Telemetry } from '../lib/types';
import { AntIcon, BookIcon, CrystalIcon, PauseIcon, PlayIcon, SlidersIcon } from './ui/icons';

interface TelemetryPillProps {
  telemetry: Telemetry;
  paused: boolean;
  speed: number;
  onTogglePause: () => void;
  onCycleSpeed: () => void;
  onGuide: () => void;
  onParams: () => void;
}

/** Top split HUD: left telemetry capsule (10 Hz, tabular-nums) and two right squircles. */
export default function TelemetryPill({
  telemetry, paused, speed, onTogglePause, onCycleSpeed, onGuide, onParams,
}: TelemetryPillProps) {
  const t = telemetry;
  const fpsColor = t.fps >= 55 ? '#00e676' : t.fps >= 40 ? '#ffa800' : '#ff0c4b';
  const ab = t.arena === 'ab-random' || t.arena === 'ab-aco';
  const rival = t.arena === 'ab-aco' ? 'ACO' : 'Random';
  return (
    <header className="hud">
      <div className="capsule glass tnum" role="status" aria-label="Colony telemetry">
        <div className="capsule-chips">
          <span className="stat" title={`${t.fps} frames per second${t.adaptive ? ' (adaptive scaler active)' : ''}`}>
            <span
              className="fps-dot"
              style={{ background: fpsColor, boxShadow: t.adaptive ? `0 0 0 3px ${fpsColor}44` : 'none' }}
            />
            <span className="stat-dim">{t.fps}</span>
          </span>
          <span className="stat">
            <AntIcon size={16} style={{ color: '#c4b5fd' }} />
            {t.alive}
            <span className="stat-dim">({t.deaths} dead)</span>
          </span>
          <span className="stat">
            <CrystalIcon size={16} style={{ color: '#00e676' }} />
            {t.food}
          </span>
          <span className="badge">GEN {t.generation}</span>
          {ab && (
            <span className="badge ab" title={`Food delivered: Trained ${t.scoreA}, ${rival} ${t.scoreB}`}>
              <span style={{ color: '#ffd24a' }}>Trained {t.scoreA}</span>
              <span className="stat-dim">:</span>
              <span style={{ color: '#fb7185' }}>{t.scoreB} {rival}</span>
            </span>
          )}
          <button type="button" className="chip-btn" onClick={onCycleSpeed} aria-label={`Simulation speed ${speed}x. Tap to change.`}>
            {speed}x
          </button>
        </div>
        <button
          type="button"
          className="play-btn"
          data-paused={paused}
          onClick={onTogglePause}
          aria-label={paused ? 'Resume simulation' : 'Pause simulation'}
        >
          {paused ? <PlayIcon size={22} /> : <PauseIcon size={22} />}
        </button>
      </div>
      <div className="squircles">
        <button type="button" className="squircle glass" onClick={onGuide} aria-label="Open field guide">
          <BookIcon size={22} />
        </button>
        <button type="button" className="squircle glass" onClick={onParams} aria-label="Open parameters and proof lab">
          <SlidersIcon size={22} />
        </button>
      </div>
    </header>
  );
}
