/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        research: {
          bg: "#0B0F17",
          panel: "#111827",
          border: "#1F2937",
          truth: "#10B981",     // Emerald (Ground truth)
          belief: "#06B6D4",    // Cyan (Agent belief)
          memory: "#8B5CF6",    // Violet (Episodic / memory)
          veto: "#F43F5E",      // Rose (Safety shield veto / danger)
          warning: "#F59E0B",   // Amber (Hazards)
        }
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      }
    },
  },
  plugins: [],
}

