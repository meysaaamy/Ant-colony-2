import type { ReactNode, SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Base({ size = 24, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

export function AntIcon(p: IconProps) {
  return (
    <Base {...p}>
      <ellipse cx="6.5" cy="12" rx="3.4" ry="2.5" />
      <ellipse cx="12.4" cy="12" rx="2" ry="1.6" />
      <circle cx="17" cy="12" r="2" />
      <path d="M11 10.4L9.2 6.5M12.6 10.4l.4-4.2M14 10.6l2.2-3.4M11 13.6l-1.8 3.9M12.6 13.6l.4 4.2M14 13.4l2.2 3.4M18.6 10.8l2.4-2M18.6 13.2l2.4 2" />
    </Base>
  );
}

export function CrystalIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M12 2.8l5 5.2-5 13.2L7 8z" />
      <path d="M7 8h10M12 2.8v18.4" />
    </Base>
  );
}

export function PlayIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M8 5.2v13.6a.6.6 0 00.9.5l11-6.8a.6.6 0 000-1L8.9 4.7a.6.6 0 00-.9.5z" fill="currentColor" stroke="none" />
    </Base>
  );
}

export function PauseIcon(p: IconProps) {
  return (
    <Base {...p}>
      <rect x="6.5" y="5" width="3.6" height="14" rx="1.2" fill="currentColor" stroke="none" />
      <rect x="13.9" y="5" width="3.6" height="14" rx="1.2" fill="currentColor" stroke="none" />
    </Base>
  );
}

export function BookIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M12 6.5C10.4 5 8 4.4 4 4.5v13.4c4-.1 6.4.5 8 2 1.6-1.5 4-2.1 8-2V4.5c-4-.1-6.4.5-8 2z" />
      <path d="M12 6.5v13.4" />
    </Base>
  );
}

export function SlidersIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M4 7h9M17 7h3M4 17h3M11 17h9" />
      <circle cx="15" cy="7" r="2" />
      <circle cx="9" cy="17" r="2" />
    </Base>
  );
}

export function FoodTrailIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M3 18c3-1 4-5 7-6s4 2 7 0 3-5 4-7" />
      <circle cx="5" cy="17.4" r="1" fill="currentColor" />
      <circle cx="12.2" cy="12.6" r="1" fill="currentColor" />
      <path d="M18.5 3.5l2 1.6-.6 2.4-2.4.5-1.5-1.9z" />
    </Base>
  );
}

export function HomeTrailIcon(p: IconProps) {
  return (
    <Base {...p}>
      <circle cx="12" cy="12" r="3.2" />
      <circle cx="12" cy="12" r="7.5" strokeDasharray="2.6 3.2" />
      <path d="M12 2v2.4M12 19.6V22M2 12h2.4M19.6 12H22" />
    </Base>
  );
}

export function AlarmIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M12 3.5l8.5 15h-17z" />
      <path d="M12 9.5v4.2" />
      <circle cx="12" cy="16.2" r="0.5" fill="currentColor" />
    </Base>
  );
}

export function PredatorIcon(p: IconProps) {
  return (
    <Base {...p}>
      <ellipse cx="11" cy="12" rx="6" ry="3.6" />
      <path d="M8 12h6M17 10.6c2 -.4 3.6-1.8 4-3.6M17 13.4c2 .4 3.6 1.8 4 3.6M7.5 9.2L5.6 5.6M11 8.4V4.6M7.5 14.8l-1.9 3.6M11 15.6v3.8" />
      <circle cx="15" cy="11" r="0.6" fill="currentColor" />
      <circle cx="15" cy="13" r="0.6" fill="currentColor" />
    </Base>
  );
}

export function WallIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M3 17.5c1.5-3 3-3.5 4.5-6s2.6-5.6 5-5.5 3 3.4 4.6 5.2S20 14 21 17.5z" />
      <path d="M6.4 17.5c1-1.6 2.2-2 3-3.6M12.5 17.5c.6-1.8 1.8-2.4 2.4-4.2" />
    </Base>
  );
}

export function EraserIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M8.5 19.5L3.6 14.6a1.4 1.4 0 010-2l8-8a1.4 1.4 0 012 0l5.8 5.8a1.4 1.4 0 010 2l-7.1 7.1z" />
      <path d="M8.4 9.8l5.8 5.8M8.5 19.5H20" />
    </Base>
  );
}

export function CloseIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M6 6l12 12M18 6L6 18" />
    </Base>
  );
}

export function ResetIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M4.5 12a7.5 7.5 0 107.5-7.5c-2.4 0-4.4 1-6 2.6" />
      <path d="M5 3.5v4h4" />
    </Base>
  );
}

export function BrainIcon(p: IconProps) {
  return (
    <Base {...p}>
      <circle cx="5" cy="7" r="1.6" />
      <circle cx="5" cy="17" r="1.6" />
      <circle cx="12" cy="5" r="1.6" />
      <circle cx="12" cy="12" r="1.6" />
      <circle cx="12" cy="19" r="1.6" />
      <circle cx="19" cy="12" r="1.6" />
      <path d="M6.5 7.4l4 3.8M6.5 16.6l4-3.8M6.6 6.6l3.8-1.2M6.6 17.4l3.8 1.2M13.5 5.8l4.2 5M13.6 12h3.8M13.5 18.2l4.2-5" />
    </Base>
  );
}

export function SkullIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M12 3.5c-4.4 0-7.5 3-7.5 7 0 2.4 1.1 4 2.8 5.2V19h9.4v-3.3c1.7-1.2 2.8-2.8 2.8-5.2 0-4-3.1-7-7.5-7z" />
      <circle cx="9" cy="11" r="1.3" />
      <circle cx="15" cy="11" r="1.3" />
      <path d="M10.5 19v-2M13.5 19v-2" />
    </Base>
  );
}

export function RadiusIcon(p: IconProps) {
  return (
    <Base {...p}>
      <circle cx="12" cy="12" r="8" strokeDasharray="2.4 3" />
      <circle cx="12" cy="12" r="2.4" />
    </Base>
  );
}

export function FlaskIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M9.5 3.5h5M10.5 3.5v5.2L5 18a1.6 1.6 0 001.4 2.5h11.2A1.6 1.6 0 0019 18l-5.5-9.3V3.5" />
      <path d="M7.6 14.5h8.8" />
    </Base>
  );
}

export function GlobeIcon(p: IconProps) {
  return (
    <Base {...p}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M3.5 12h17M12 3.5c2.6 2.4 3.8 5.2 3.8 8.5s-1.2 6.1-3.8 8.5c-2.6-2.4-3.8-5.2-3.8-8.5s1.2-6.1 3.8-8.5z" />
    </Base>
  );
}

export function ChartIcon(p: IconProps) {
  return (
    <Base {...p}>
      <path d="M4 4v16h16" />
      <path d="M7 15.5l3.6-4.2 3 2.4L19 7" />
    </Base>
  );
}

export function BrandMark({ size = 36 }: { size?: number } = {}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  );
}

const ICON_MAP: Record<string, React.ComponentType<IconProps>> = {
  ant: AntIcon,
  crystal: CrystalIcon,
  play: PlayIcon,
  pause: PauseIcon,
  book: BookIcon,
  sliders: SlidersIcon,
  'food-trail': FoodTrailIcon,
  'home-trail': HomeTrailIcon,
  alarm: AlarmIcon,
  predator: PredatorIcon,
  wall: WallIcon,
  eraser: EraserIcon,
  close: CloseIcon,
  reset: ResetIcon,
  brain: BrainIcon,
  skull: SkullIcon,
  radius: RadiusIcon,
  flask: FlaskIcon,
  globe: GlobeIcon,
  chart: ChartIcon,
};

export function Icon({ name, size = 16 }: { name: string; size?: number }) {
  const Comp = ICON_MAP[name] || Base;
  return <Comp size={size} />;
}
