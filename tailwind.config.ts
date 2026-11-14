import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      borderRadius: {
        sm: '12px',
        md: '20px',
        lg: '28px',
        squircle: '14px',
        peek: '22px',
        dock: '24px',
      },
      colors: {
        abyss: '#020307',
        basalt: '#0d1322',
        rimteal: '#2dd4bf',
        rimsky: '#38bdf8',
        emeraldcore: '#00e676',
        facet: '#a7f3d0',
        amber5: '#ffa800',
        cyan5: '#00e1ff',
        crimson5: '#ff0c4b',
        violet5: '#af32ff',
        viridian5: '#50ff64',
      },
      fontFamily: {
        display: ['var(--font-fraunces)', 'Georgia', 'serif'],
        sans: ['var(--font-jakarta)', 'system-ui', 'sans-serif'],
      },
      transitionTimingFunction: {
        sanctuary: 'cubic-bezier(0.2, 0.8, 0.2, 1)',
      },
    },
  },
};

export default config;
