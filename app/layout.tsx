import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Formica Sanctuary · Neuroevolved Ant Colony',
  description:
    'Astra visual identity with Fable complete nav bar: top HUD + bottom 3-snap cockpit + 6-tool dock. 3,000 ants, five pheromones, shared colony brain.',
  icons: { icon: '/logo.svg' },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: 'cover',
  themeColor: '#020307',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="font-sans antialiased">
      <body>{children}</body>
    </html>
  );
}
