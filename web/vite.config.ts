import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths so dist/ can be hosted as plain files from any folder.
  base: "./",
  // The app imports ../params.json (repo root) so edits there show up without a wasm rebuild.
  server: { fs: { allow: [".."] }, port: 5173, strictPort: true },
  build: { outDir: "dist", target: "es2022" },
});
