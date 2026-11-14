'use client';

import { useEffect, useRef } from 'react';
import { CHANNEL_CSS } from '../lib/types';
import { CloseIcon } from './ui/icons';

interface GuideEntry {
  title: string;
  nature: string;
  sandbox: string;
  color: string;
}

const ENTRIES: readonly GuideEntry[] = [
  {
    title: 'Trail pheromones',
    color: CHANNEL_CSS[0],
    nature:
      'Foragers that find food lay a volatile trail on the way home. Each ant that follows and succeeds reinforces it; unrewarded trails evaporate. Short routes are refreshed faster, so the colony converges on them.',
    sandbox:
      'Carriers secrete Solar Amber foodTrail and foragers secrete Abyssal Cyan homeGuide. Where both overlap they polymerise into pearl-white highways that decay 35% slower.',
  },
  {
    title: 'Path integration',
    color: CHANNEL_CSS[1],
    nature:
      'Desert ants keep a running home vector by integrating every step and turn, and can walk straight back to the nest across featureless ground.',
    sandbox:
      'Every ant carries (homeDx, homeDy). The shared brain reads it egocentrically; mute it in the Proof Lab to see how much the colony depends on it.',
  },
  {
    title: 'Alarm and defence',
    color: CHANNEL_CSS[2],
    nature:
      'Alarm pheromones are small, fast-diffusing molecules. Workers scatter while soldiers move toward the source.',
    sandbox:
      'Ants within 75 px of an Abyssal Stalker release a +80 Crimson burst that decays 3.2x faster than food trail. Foragers flee down the gradient; Wardens climb it and swarm the predator.',
  },
  {
    title: 'Necrophoresis',
    color: '#8f8172',
    nature: 'Dead nestmates are detected by their chemistry and carried to refuse piles, limiting disease.',
    sandbox:
      'Ants die of starvation or old age and leave a chitin husk that leaks weak alarm. Wardens haul husks to a graveyard placed away from food routes.',
  },
  {
    title: 'Tandem running and trophallaxis',
    color: CHANNEL_CSS[3],
    nature:
      'Some species lead a single recruit to a find, and most share liquid food mouth to mouth, pooling the colony stomach.',
    sandbox:
      'Returning Scouts hand crystal coordinates to nearby foragers with an Electric Violet scoutPulse. Energy-rich ants top up peers that fall under 30% ATP.',
  },
  {
    title: 'Dead-end sealing',
    color: CHANNEL_CSS[4],
    nature: 'Pharaoh ants mark unrewarding branches with a repellent "no entry" pheromone.',
    sandbox:
      'Ants that stall in a trap or reach an exhausted crystal turn around and lay Acid Viridian noEntry, which chemically dissolves stale foodTrail.',
  },
];

const SHORTCUTS: ReadonlyArray<{ keys: string[]; text: string }> = [
  { keys: ['1', '2', '3', '4', '5', '6'], text: 'Select Food, Home, Alarm, Crystal, Wall, Eraser' },
  { keys: ['Space'], text: 'Play or pause' },
  { keys: ['[', ']'], text: 'Shrink or grow the brush' },
  { keys: ['Esc'], text: 'Close this guide or collapse the sheet' },
];

function MixDiagram() {
  return (
    <svg viewBox="0 0 320 150" width="100%" role="img" aria-label="How the five pheromone colours mix">
      <rect width="320" height="150" rx="12" fill="#020307" />
      <g style={{ mixBlendMode: 'screen' }}>
        <circle cx="70" cy="62" r="34" fill="rgb(255,168,0)" fillOpacity="0.8" />
        <circle cx="112" cy="62" r="34" fill="rgb(0,225,255)" fillOpacity="0.8" />
        <circle cx="196" cy="56" r="28" fill="rgb(255,12,75)" fillOpacity="0.8" />
        <circle cx="232" cy="56" r="28" fill="rgb(80,255,100)" fillOpacity="0.75" />
        <circle cx="268" cy="100" r="24" fill="rgb(175,50,255)" fillOpacity="0.8" />
      </g>
      <ellipse cx="91" cy="62" rx="9" ry="24" fill="#f5faff" />
      <ellipse cx="214" cy="56" rx="7" ry="18" fill="#ecd82e" />
      <text x="91" y="116" textAnchor="middle" fontSize="10" fill="#f5faff">pearl highway core</text>
      <text x="91" y="130" textAnchor="middle" fontSize="9" fill="#5eead4">turquoise edges</text>
      <text x="214" y="104" textAnchor="middle" fontSize="10" fill="#ecd82e">sulfur-gold hazard</text>
      <text x="268" y="140" textAnchor="middle" fontSize="9" fill="#d8b4fe">scout violet</text>
      <text x="160" y="16" textAnchor="middle" fontSize="10" fill="#9aa8bf">rgb = 1 - exp(-gamma * sum(w * C * RGB)) + interference</text>
    </svg>
  );
}

function BrainDiagram() {
  const ins = 10;
  const hid = 8;
  const outs = 9;
  const y = (i: number, n: number): number => 22 + (i * 110) / (n - 1);
  const lines: Array<{ x1: number; y1: number; x2: number; y2: number; k: string }> = [];
  for (let i = 0; i < ins; i++) for (let h = 0; h < hid; h++) if ((i * 3 + h) % 3 === 0) lines.push({ x1: 70, y1: y(i, ins), x2: 160, y2: y(h, hid), k: `a${i}-${h}` });
  for (let h = 0; h < hid; h++) for (let o = 0; o < outs; o++) if ((h + o * 2) % 3 === 0) lines.push({ x1: 160, y1: y(h, hid), x2: 250, y2: y(o, outs), k: `b${h}-${o}` });
  return (
    <svg viewBox="0 0 320 150" width="100%" role="img" aria-label="Shared colony brain: 19 inputs, 16 hidden units, 9 outputs">
      <rect width="320" height="150" rx="12" fill="#020307" />
      {lines.map((l) => (
        <line key={l.k} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="#2dd4bf" strokeOpacity="0.28" strokeWidth="1" />
      ))}
      <path d="M250 132C290 146 30 146 70 132" stroke="#af32ff" strokeOpacity="0.7" strokeWidth="1.2" fill="none" strokeDasharray="3 4" />
      {Array.from({ length: ins }, (_, i) => (
        <circle key={`i${i}`} cx="70" cy={y(i, ins)} r="3.4" fill={i < 5 ? CHANNEL_CSS[i] : '#9aa8bf'} />
      ))}
      {Array.from({ length: hid }, (_, i) => (
        <circle key={`h${i}`} cx="160" cy={y(i, hid)} r="4" fill="#0d1322" stroke="#38bdf8" strokeWidth="1.4" />
      ))}
      {Array.from({ length: outs }, (_, i) => (
        <circle key={`o${i}`} cx="250" cy={y(i, outs)} r="3.4" fill={i >= 2 && i <= 6 ? CHANNEL_CSS[i - 2] : '#ffd24a'} />
      ))}
      <text x="12" y="14" fontSize="10" fill="#9aa8bf">19 inputs</text>
      <text x="136" y="14" fontSize="10" fill="#9aa8bf">16 tanh</text>
      <text x="258" y="14" fontSize="10" fill="#9aa8bf">9 outputs</text>
      <text x="262" y="60" fontSize="9" fill="#64728a">steer, speed</text>
      <text x="262" y="82" fontSize="9" fill="#64728a">5 glands</text>
      <text x="262" y="128" fontSize="9" fill="#d8b4fe">memA, memB</text>
    </svg>
  );
}

export default function FieldGuideModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);
  if (!open) return null;
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div
        className="modal glass"
        role="dialog"
        aria-modal="true"
        aria-labelledby="guide-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-2">
          <div>
            <h2 id="guide-title" className="font-display text-[22px] leading-7 font-semibold">Field Guide</h2>
            <p className="section-note">What real ants do, and what this sandbox models.</p>
          </div>
          <button ref={closeRef} type="button" className="squircle" onClick={onClose} aria-label="Close field guide">
            <CloseIcon size={22} />
          </button>
        </div>
        <div className="modal-body">
          <div className="guide-grid">
            <figure className="card">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/assets/diagram-double-bridge.svg" alt="Deneubourg's double bridge: the short branch is reinforced first" width={320} height={150} style={{ width: '100%', height: 'auto' }} />
              <figcaption className="section-note">Deneubourg&apos;s double bridge: positive feedback selects the short branch.</figcaption>
            </figure>
            <figure className="card">
              <MixDiagram />
              <figcaption className="section-note">Five channels are mixed per cell by concentration, then pairwise interference tints the result.</figcaption>
            </figure>
            <figure className="card">
              <BrainDiagram />
              <figcaption className="section-note">Every trained ant runs the same recurrent network. Evolution scores whole colonies, not individuals.</figcaption>
            </figure>
            <div className="card">
              <h3 className="section-title">Shortcuts</h3>
              {SHORTCUTS.map((s) => (
                <div key={s.text} className="flex items-center gap-2 text-[12px] leading-[18px]">
                  <span className="flex flex-none flex-wrap gap-1">
                    {s.keys.map((k) => (
                      <kbd key={k} className="kbd">{k}</kbd>
                    ))}
                  </span>
                  <span className="section-note">{s.text}</span>
                </div>
              ))}
              <div className="flex gap-2 pt-1">
                {['food-trail', 'home-trail', 'alarm', 'food-cluster', 'wall', 'eraser'].map((n) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img key={n} src={`/icons/tool-${n}.svg`} alt="" width={24} height={24} />
                ))}
              </div>
            </div>
          </div>
          {ENTRIES.map((e) => (
            <section key={e.title} className="flex flex-col gap-2">
              <h3 className="section-title flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: e.color }} />
                {e.title}
              </h3>
              <div className="guide-grid">
                <div className="card">
                  <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-[#a7f3d0]">In nature</span>
                  <p className="section-note">{e.nature}</p>
                </div>
                <div className="card">
                  <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-[#9af3ff]">In this sandbox</span>
                  <p className="section-note">{e.sandbox}</p>
                </div>
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
