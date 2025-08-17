/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'tmdb-blue': '#0d253f',
        'tmdb-light-blue': '#01b4e4',
        'tmdb-green': '#90cea1',
        // Dark theme colors
        'dark-bg': '#111827',
        'dark-card': '#1f2937',
        'dark-border': '#374151',
      },
    },
  },
  plugins: [],
}
