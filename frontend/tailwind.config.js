/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#eaebef",
        surface: "#ffffff",
        primary: { DEFAULT: "#09568c", hover: "#073f66" },
        muted: "#a1a8ae",
        ink: "#0f2233",
        danger: "#b3261e",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
