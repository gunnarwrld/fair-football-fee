import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `next build` also writes .next/standalone: a minimal server.js plus only the node_modules it
  // needs, which is what frontend/Dockerfile ships
  output: "standalone",
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
