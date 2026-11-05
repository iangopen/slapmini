import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  // Relative asset paths so dist/ can be hosted as plain files from any folder (GitHub Pages: /REPO/).
  base: "./",
  // The app imports ../params.json (repo root) so edits there show up without a wasm rebuild.
  server: { fs: { allow: [".."] }, port: 5173, strictPort: true },
  build: {
    outDir: "dist",
    target: "es2022",
    rollupOptions: {
      // index.html = the game (menu, match vs bot); lab.html = the feel lab (Practice).
      input: { index: resolve(__dirname, "index.html"), lab: resolve(__dirname, "lab.html") },
    },
  },
  // onnxruntime-web: Vite emits its wasm runtime (ort-wasm-simd-threaded*.wasm) into dist/assets/ with a
  // relative URL; the dev server must not pre-bundle it (that breaks the import.meta.url lookup).
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
