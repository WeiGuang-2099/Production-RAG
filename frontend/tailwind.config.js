/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#FAF9F5",
        surface: "#ffffff",
        sunken: "#F5F4EF",
        ink: "#1C1917",
        muted: "#78716C",
        line: "#E7E5E0",
        primary: { DEFAULT: "#B45309", hover: "#92400E" },
        highlight: { DEFAULT: "#FEF3C7", ink: "#92400E" },
        danger: "#B3261E",
        success: "#15803D",
      },
      fontFamily: {
        sans: ["'Inter Variable'", "system-ui", "sans-serif"],
        serif: ["'Source Serif 4 Variable'", "Georgia", "serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
