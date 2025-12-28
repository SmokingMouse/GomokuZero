/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{js,jsx}", "./components/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101217",
        chalk: "#f6f3ee",
        ember: "#e07a2f",
        forest: "#234338"
      },
      fontFamily: {
        display: ["\"Space Grotesk\"", "system-ui", "sans-serif"],
        mono: ["\"IBM Plex Mono\"", "ui-monospace", "SFMono-Regular"]
      }
    }
  },
  plugins: []
};
