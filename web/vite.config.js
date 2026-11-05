import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/infer": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/doctor-reviews": "http://127.0.0.1:8000",
    },
  },
});
