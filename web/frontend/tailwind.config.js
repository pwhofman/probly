import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#faf9f5',
        panel: '#f4f2ec',
        rule: '#e8e5dc',
        ink: '#1f1e1a',
        muted: '#8b8778',
        accent: '#cc785c',
      },
      fontFamily: {
        serif: ['"Tiempos Headline"', '"Source Serif Pro"', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [typography],
};
