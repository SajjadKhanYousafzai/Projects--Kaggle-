/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        compliant: '#16a34a',
        violation: '#dc2626',
        neutral:   '#6b7280',
      },
    },
  },
  plugins: [],
}
