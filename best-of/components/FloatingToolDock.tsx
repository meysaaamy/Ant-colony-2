/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2.zip/components/FloatingToolDock.tsx)
 * Why best: رابط الفابلهی séd interacting با sim: ۶ ابزار، صفحه ۳냅، HUD،_top pills
 * Borrow from other zips: mobile: BioSlider (دابل‌تاپ reset)؛ astramax: D3 dispersion graph، تجربیات /api/health
 * Notes: براي ادجام به best-of/components/FloatingToolDock.tsx
 * Status: verbatim copy — adapt import paths when merging into app/
 */

'use client';

import { useState } from 'react';
import type { ComponentType, CSSProperties, SVGProps } from 'react';
import { CHANNEL_CSS, CHANNEL_NAMES } from '../lib/types';
import type { SubMode, ToolId } from '../lib/types';
import {
  AlarmIcon, CrystalIcon, EraserIcon, FoodTrailIcon, HomeTrailIcon, PredatorIcon, RadiusIcon, WallIcon,
} from './ui/icons';

type IconType = ComponentType<SVGProps<SVGSVGElement> & { size?: number }>;

export interface ToolDef {
  id: ToolId;
  label: string;
  color: string;
  hint: string;
  Icon: IconType;
}

export const TOOLS: readonly ToolDef[] = [
  { id: 'food-trail', label: 'Food', color: CHANNEL_CSS[0], hint: 'Paint pheromone', Icon: FoodTrailIcon },
  { id: 'home-trail', label: 'Home', color: CHANNEL_CSS[1], hint: 'Paint pheromone', Icon: HomeTrailIcon },
  { id: 'alarm', label: 'Alarm', color: CHANNEL_CSS[2], hint: 'Paint pheromone or spawn a predator', Icon: AlarmIcon },
  { id: 'crystal', label: 'Crystal', color: '#00e676', hint: 'Drop emerald food caches', Icon: CrystalIcon },
  { id: 'wall', label: 'Wall', color: '#38bdf8', hint: 'Sculpt terraced basalt', Icon: WallIcon },
  { id: 'eraser', label: 'Eraser', color: '#e2e8f0', hint: 'Erase trails, basalt, caches and predators', Icon: EraserIcon },
];

export function isPheromoneTool(tool: ToolId): boolean {
  return tool === 'food-trail' || tool === 'home-trail' || tool === 'alarm';
}

interface DockProps {
  tool: ToolId;
  subMode: SubMode;
  radius: number;
  onTool: (tool: ToolId) => void;
  onSubMode: (sub: SubMode) => void;
  onRadius: (r: number) => void;
}

const SUBS: readonly SubMode[] = [0, 1, 2, 3, 4, 5];

/** Tier 2 contextual pills (5 pheromones + predator, radius popover) and the Tier 3 six-tool jewel dock. */
export default function FloatingToolDock({ tool, subMode, radius, onTool, onSubMode, onRadius }: DockProps) {
  const [radiusOpen, setRadiusOpen] = useState(false);
  const active = TOOLS.find((t) => t.id === tool) ?? TOOLS[0];
  const pher = isPheromoneTool(tool);
  return (
    <div className="dock-wrap">
      <div className="sub-row glass">
        <span className="sub-label" style={{ color: active.color }}>{active.label}</span>
        <div className="pill-row" role="radiogroup" aria-label="Brush sub-mode">
          {pher ? (
            SUBS.map((s) => {
              const color = s === 5 ? CHANNEL_CSS[2] : CHANNEL_CSS[s];
              const style = { '--pill-color': color } as CSSProperties;
              return (
                <button
                  key={s}
                  type="button"
                  role="radio"
                  aria-checked={subMode === s}
                  className="pill"
                  data-active={subMode === s}
                  style={style}
                  onClick={() => onSubMode(s)}
                >
                  {s === 5 ? <PredatorIcon size={16} style={{ color }} /> : <span className="dot" />}
                  {s === 5 ? 'Spawn Predator' : CHANNEL_NAMES[s]}
                </button>
              );
            })
          ) : (
            <span className="pill" style={{ cursor: 'default' }}>{active.hint}</span>
          )}
        </div>
        <button
          type="button"
          className="pill"
          data-active={radiusOpen}
          aria-expanded={radiusOpen}
          aria-label={`Brush radius ${radius} pixels`}
          onClick={() => setRadiusOpen((o) => !o)}
        >
          <RadiusIcon size={16} />
          <span className="tnum">{radius}</span>
        </button>
        {radiusOpen && (
          <div className="popover glass">
            <label className="field">
              <span>Brush radius</span>
              <output className="tnum">{radius}px</output>
              <input
                type="range"
                min={8}
                max={64}
                step={2}
                value={radius}
                onChange={(e) => onRadius(Number(e.target.value))}
              />
            </label>
          </div>
        )}
      </div>
      <nav className="dock glass" aria-label="Tools">
        {TOOLS.map((t, i) => {
          const style = { '--slot-color': t.color } as CSSProperties;
          return (
            <button
              key={t.id}
              type="button"
              className="slot"
              data-active={tool === t.id}
              style={style}
              aria-pressed={tool === t.id}
              aria-label={`${t.label} tool, shortcut ${i + 1}`}
              title={`${t.label} (${i + 1})`}
              onClick={() => onTool(t.id)}
            >
              <t.Icon size={24} />
              <span className="slot-index tnum">{i + 1}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}
