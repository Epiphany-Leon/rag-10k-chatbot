import type { NextConfig } from "next";

// Two modes:
//  - dev / default: proxy /api/* to the FastAPI backend (so the browser sees one
//    same-origin host during local development).
//  - BUILD_EXPORT=1: emit a static site to web/out/ which FastAPI serves in
//    production (no Node needed on the server). /api is routed by the same
//    FastAPI process, so no rewrite is required.
const API = process.env.API_URL || "http://127.0.0.1:8600";
const isExport = process.env.BUILD_EXPORT === "1";

const nextConfig: NextConfig = isExport
  ? { output: "export", images: { unoptimized: true } }
  : {
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
      },
    };

export default nextConfig;
