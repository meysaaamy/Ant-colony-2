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
