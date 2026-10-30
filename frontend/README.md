# fair-football-fee frontend

Next.js 16 (App Router, Cache Components) UI for the fair-football-fee API: player search, a valuation page
(estimate vs Transfermarkt value, prediction interval, SHAP drivers) and a what-if panel.

## Run

The API must be running first (`make api` from the repo root, after `make db` and `make register`).

```bash
npm install
npm run dev        # http://localhost:3000
```

`NEXT_PUBLIC_API_BASE_URL` points at the API (default `http://localhost:8000/v1`). It is inlined at build time.

## Checks

```bash
npm run typecheck  # route types + tsc
npm run lint
npm test           # vitest: formatters and chart math
npm run build
npm run api:types  # regenerate src/lib/api-types.ts from openapi.json
```

`openapi.json` is the API contract, written from the Python code by `make api-types` at the repo root (which
also runs `api:types`). Both files are committed; CI regenerates them and fails if either differs.

## Docker

`Dockerfile` builds the standalone server (`output: "standalone"` in `next.config.ts`) and runs it as the
`node` user. `NEXT_PUBLIC_API_BASE_URL` is a build argument because Next.js inlines it into the browser
bundle: `docker compose --profile app up` from the repo root (`make up`) passes it from `.env`.

## Design notes

- **Types come from the API.** `src/lib/api-types.ts` is generated from FastAPI's OpenAPI schema (`openapi.json`),
  so a backend change that breaks the UI fails `npm run typecheck`, not the browser.
- **The browser calls the API directly.** The what-if rate limit is per client IP; routing calls through Next.js
  server code would put every visitor in one bucket. Pages are static shells (the player page reads its route
  param inside `<Suspense>`, as Cache Components requires) and data loads with SWR after hydration.
- **The what-if ranges come from `GET /v1/model`**, the same bounds the API enforces, so the sliders can't offer a
  value the API will refuse.
- **Charts follow the dataviz guidance:** colors from a validated palette in both light and dark mode
  (diverging blue/red for drivers, a one-hue ramp for the interval), log-scale bars so +100% and −50% are the
  same length, legends that carry the numbers, and a driver chart that is also a table.
- **Money is formatted by hand** (`src/lib/format.ts`): `Intl`'s compact notation differs between runtimes
  (Node prints "€22.4M", Chrome "€22.4m").
