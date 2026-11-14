/*
 * BEST-OF SOURCE FILE
 * Source zip: fable 5.2.zip (fable_5.2.zip/components/ToastOverlay.tsx)
 * Why best: رابط الفابلهی séd interacting با sim: ۶ ابزار، صفحه ۳냅، HUD،_top pills
 * Borrow from other zips: mobile: BioSlider (دابل‌تاپ reset)؛ astramax: D3 dispersion graph، تجربیات /api/health
 * Notes: براي ادجام به best-of/components/ToastOverlay.tsx
 * Status: verbatim copy — adapt import paths when merging into app/
 */

'use client';

import type { ToastMessage } from '../lib/types';

const TONE: Record<ToastMessage['tone'], string> = {
  info: '#00e1ff',
  good: '#00e676',
  warn: '#ffa800',
};

export default function ToastOverlay({ toasts }: { toasts: ToastMessage[] }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className="toast glass">
          <i style={{ background: TONE[t.tone] }} />
          <span>{t.text}</span>
        </div>
      ))}
    </div>
  );
}
