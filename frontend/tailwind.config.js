/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        espresso: {
          950: "#120D0A",
          900: "#1C1510",
          800: "#2A2019",
          700: "#3D2E24",
        },
        burgundy: {
          600: "#7B1E34",
          700: "#5C1425",
          800: "#3D0D18",
        },
        cream: {
          100: "#FAF7F0",
          200: "#EFE7DA",
          300: "#D4C5A9",
          400: "#B8A484",
        },
        forensic: {
          900: "#0b0f19",
          850: "#101626",
          800: "#161e32",
          700: "#222e4d",
          600: "#32446e",
          500: "#49629b",
          accent: "#3b82f6",
          teal: "#0d9488",
          emerald: "#10b981",
          amber: "#f59e0b",
          rose: "#f43f5e",
        },
      },
    },
  },
  plugins: [],
};
