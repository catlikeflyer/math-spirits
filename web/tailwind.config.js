/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        academic: {
          bg: '#131416',
          paper: '#fcfbf9',
          card: '#1a1c20',
          cardHover: '#212429',
          cardLight: '#ffffff',
          border: '#2a2d35',
          borderLight: '#e5e2db',
          ink: '#eceae3',
          inkLight: '#1c1917',
          muted: '#8e8b83',
          mutedLight: '#68645e',
          accent: '#3b82f6',
          oxford: '#1e3a8a',
        },
      },
      fontFamily: {
        serif: ['Newsreader', 'Georgia', 'Cambria', 'Times New Roman', 'serif'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
};
