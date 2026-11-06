/** @type {import('tailwindcss').Config} */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: token('bg'),
        panel: token('panel'),
        raised: token('raised'),
        sunken: token('sunken'),
        line: token('line'),
        fg: token('fg'),
        muted: token('muted'),
        faint: token('faint'),
        accent: token('accent'),
        'accent-fg': token('accent-fg'),
        ok: token('ok'),
        warn: token('warn'),
        err: token('err'),
        info: token('info'),
      },
      fontFamily: {
        sans: ['"Segoe UI Variable Text"', '"Segoe UI"', 'system-ui', '-apple-system', 'Roboto', 'sans-serif'],
        mono: ['"Cascadia Code"', '"Cascadia Mono"', 'Consolas', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', '1rem'],
      },
      keyframes: {
        'fade-in': { from: { opacity: '0', transform: 'translateY(2px)' }, to: { opacity: '1', transform: 'none' } },
        pulse2: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.35' } },
      },
      animation: {
        'fade-in': 'fade-in 120ms ease-out',
        pulse2: 'pulse2 1.4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
