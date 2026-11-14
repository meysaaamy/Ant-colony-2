/*
 * BEST-OF SOURCE FILE
 * Source zip: mobile-bioluminescent-ui-design.zip (mobile-bioluminescent-ui-design/src/components/ui/BioSlider.tsx)
 * Why best: اسلایدر موبایل با دابل‌تاپ به reset، technicalTerm/helperText، رنگ‌های accent/core/ambient، حالت compact
 * Borrow from other zips: fable: ParameterSheet/FloatingToolDock
 * Notes: مرجع رابط موبایل — در best-of/refs/
 * Status: verbatim copy — adapt import paths when merging into app/
 */

"use client";

import React, { useRef } from "react";

export interface BioSliderProps {
  id: string;
  label: string;
  technicalTerm?: string;
  helperText?: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  formatValue?: (val: number) => string;
  accentColor?: string;
  coreColor?: string;
  ambientColor?: string;
  onChange: (value: number) => void;
  compact?: boolean;
}

export function BioSlider({
  id,
  label,
  technicalTerm,
  helperText,
  value,
  defaultValue,
  min,
  max,
  step,
  unit = "",
  formatValue,
  accentColor = "#ffae00",
  coreColor = "#fff3b0",
  ambientColor = "rgba(255, 174, 0, 0.42)",
  onChange,
  compact = false,
}: BioSliderProps) {
  const lastTapRef = useRef<number>(0);

  const pct = Math.min(100, Math.max(0, ((value - min) / Math.max(1e-6, max - min)) * 100));
  const defaultPct = Math.min(
    100,
    Math.max(0, ((defaultValue - min) / Math.max(1e-6, max - min)) * 100)
  );

  const displayText = formatValue
    ? formatValue(value)
    : `${Number.isInteger(step) ? Math.round(value) : value.toFixed(2)}${unit ? ` ${unit}` : ""}`;

  const handleDoubleTapCheck = () => {
    const now = performance.now();
    if (now - lastTapRef.current < 300) {
      onChange(defaultValue);
      lastTapRef.current = 0;
    } else {
      lastTapRef.current = now;
    }
  };

  return (
    <div className={compact ? "space-y-1" : "space-y-1.5"}>
      <div className="flex items-baseline justify-between gap-2">
        <label
          htmlFor={id}
          onDoubleClick={() => onChange(defaultValue)}
          className="text-[13px] font-medium text-slate-200 leading-tight cursor-pointer"
          title="Double-tap to reset to default"
        >
          {label}
          {technicalTerm && (
            <span className="ml-1.5 text-[11px] font-normal text-slate-400">
              ({technicalTerm})
            </span>
          )}
        </label>
        <span
          className="tabular-nums text-[12px] font-semibold shrink-0"
          style={{ color: coreColor }}
        >
          {displayText}
        </span>
      </div>

      <div
        className="bio-slider-wrap"
        onPointerUp={handleDoubleTapCheck}
        style={
          {
            "--slider-accent": accentColor,
            "--slider-core": coreColor,
            "--slider-ambient": ambientColor,
          } as React.CSSProperties
        }
      >
        {/* 6px Dark Glass Track */}
        <div className="pointer-events-none absolute inset-x-0 h-[6px] rounded-full bg-slate-950/90 shadow-[inset_0_1px_2px_rgba(0,0,0,0.85),0_0_0_1px_rgba(255,255,255,0.08)] overflow-hidden">
          <div
            className="h-full rounded-full transition-[width] duration-75"
            style={{
              width: `${pct}%`,
              background: `linear-gradient(90deg, ${accentColor}99 0%, ${accentColor} 65%, ${coreColor} 100%)`,
              boxShadow: `0 0 12px ${ambientColor}`,
            }}
          />
        </div>

        {/* Default-value tick mark */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-[2px] h-3 rounded-full bg-white/30 z-[1]"
          style={{ left: `${defaultPct}%` }}
        />

        {/* Invisible 44px touch hit zone around native range input */}
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          aria-valuetext={displayText}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="bio-slider h-[44px]"
        />
      </div>

      {helperText && !compact && (
        <p className="text-[11px] leading-snug text-slate-400/90">
          {helperText}
        </p>
      )}
    </div>
  );
}
