// k6 load test of the fair-football-fee API. Run it with `make loadtest` (scripts/loadtest.sh): that starts
// the API container, runs this script three times (see RUNS there) and writes reports/phase7_load_test.md
// from the JSON each run saves.
//
// Three request types run at the same time, each at a fixed arrival rate. It's an open model: new requests
// keep arriving even when the API slows down, as real users would, so overload shows up as latency and
// refusals instead of being hidden by a client that waits.
//   valuation  GET  /players/{id}/valuation  every servable player, warmed in setup(): Redis cache hits
//   search     GET  /players?q=               Postgres trigram search, never cached
//   whatif     POST /valuations               a new random age every time: cache miss, model + TreeSHAP
//
// EXPECT sets the pass/fail thresholds of a run:
//   fast    no errors, at most 1% shed with 503 "busy", and the p95 budgets below (normal operation;
//           with 16 requests per worker, a short stall at this load can shed a few requests, by design)
//   served  every request answered 200, however slowly (Redis down, load within uncached capacity)
//   shed    every response is a 200 or a 503 "busy" (never a 500), and none waits past the 2 s database
//           pool timeout plus its own work (Redis down, load above uncached capacity)

import http from "k6/http";
import { Rate, Trend } from "k6/metrics";

const BASE = __ENV.BASE_URL || "http://api:8000/v1";
const RUN = __ENV.RUN || "redis_up";
const EXPECT = __ENV.EXPECT || "fast";
const DURATION = __ENV.DURATION || "60s";
const RATES = {
  valuation: Number(__ENV.VALUATION_RATE || 200),
  search: Number(__ENV.SEARCH_RATE || 40),
  whatif: Number(__ENV.WHATIF_RATE || 20),
};
const NAMES = Object.keys(RATES);
// Name fragments to discover player ids through the API itself (ids are internal, never hard-coded)
const DISCOVERY_TERMS = ["son", "mar", "ali", "ben", "dav", "and", "ric", "tom", "jam", "lew", "kev", "mat",
  "dan", "jo", "ch", "el", "an", "er", "in", "ar", "ro", "la", "ma", "ne"];
const SEARCH_TERMS = ["son", "mar", "ali", "ben", "dav", "and", "ric", "tom", "jam", "lew", "kev", "mat"];
const P95_BUDGET_MS = { valuation: 50, search: 100, whatif: 300 };
const SHED_P99_MS = 3000;
const FAST_MAX_SHED = 0.01;

const served = Object.fromEntries(NAMES.map((n) => [n, new Trend(`${n}_ms`, true)])); // 200s only
const anyLatency = Object.fromEntries(NAMES.map((n) => [n, new Trend(`${n}_any_ms`, true)])); // every response
const ok = Object.fromEntries(NAMES.map((n) => [n, new Rate(`${n}_ok`)]));
const shed = Object.fromEntries(NAMES.map((n) => [n, new Rate(`${n}_shed`)]));
const failed = Object.fromEntries(NAMES.map((n) => [n, new Rate(`${n}_error`)]));
const cacheHit = Object.fromEntries(NAMES.map((n) => [n, new Rate(`${n}_cache_hit`)]));

function thresholdsFor(n) {
  if (EXPECT === "fast") {
    return [[`${n}_error`, ["rate==0"]], [`${n}_shed`, [`rate<${FAST_MAX_SHED}`]],
      [`${n}_ms`, [`p(95)<${P95_BUDGET_MS[n]}`]]];
  }
  if (EXPECT === "served") return [[`${n}_ok`, ["rate>0.999"]]];
  return [[`${n}_error`, ["rate==0"]], [`${n}_any_ms`, [`p(99)<${SHED_P99_MS}`]]];
}

export const options = {
  scenarios: Object.fromEntries(NAMES.map((n) => [n, {
    executor: "constant-arrival-rate",
    exec: n,
    rate: RATES[n],
    timeUnit: "1s",
    duration: DURATION,
    preAllocatedVUs: Math.max(5, RATES[n]),
    maxVUs: Math.max(50, 4 * RATES[n]),
  }])),
  thresholds: Object.fromEntries(NAMES.flatMap(thresholdsFor)),
  summaryTrendStats: ["count", "med", "p(95)", "p(99)", "max"],
};

const pick = (items) => items[Math.floor(Math.random() * items.length)];

function record(name, res) {
  const busy = res.status === 503 && res.body && res.body.includes('"busy"');
  anyLatency[name].add(res.timings.duration);
  ok[name].add(res.status === 200);
  shed[name].add(busy);
  failed[name].add(res.status !== 200 && !busy);
  if (res.status === 200) {
    served[name].add(res.timings.duration);
    cacheHit[name].add(res.headers["X-Cache"] === "HIT");
  }
}

export function setup() {
  const ids = new Set();
  for (const q of DISCOVERY_TERMS) {
    const res = http.get(`${BASE}/players?q=${q}&limit=25`);
    for (const p of res.status === 200 ? res.json("results") : []) if (p.has_valuation) ids.add(p.id);
  }
  const players = [...ids];
  if (players.length === 0) throw new Error(`no servable players found at ${BASE}: run make db and make register`);
  for (const id of players) http.get(`${BASE}/players/${id}/valuation`); // fill the cache (no-op without Redis)
  return { players };
}

export function valuation(data) {
  record("valuation", http.get(`${BASE}/players/${pick(data.players)}/valuation`));
}

export function search() {
  record("search", http.get(`${BASE}/players?q=${pick(SEARCH_TERMS)}`));
}

export function whatif(data) {
  // Any age inside the trained 17–44 range is a valid scenario; two decimals make repeats unlikely
  const scenario = { age_years: Math.round(1700 + Math.random() * 2000) / 100 };
  const body = JSON.stringify({ player_id: pick(data.players), scenario });
  record("whatif", http.post(`${BASE}/valuations`, body, { headers: { "Content-Type": "application/json" } }));
}

export function handleSummary(data) {
  const metric = (name) => data.metrics[name] || { values: {}, thresholds: {} };
  const result = {
    meta: {
      run: RUN,
      expect: EXPECT,
      redis: __ENV.REDIS || null,
      duration: DURATION,
      rates_per_second: RATES,
      p95_budget_ms: EXPECT === "fast" ? P95_BUDGET_MS : null,
      shed_p99_ms: EXPECT === "shed" ? SHED_P99_MS : null,
      players: data.setup_data ? data.setup_data.players.length : 0,
      api_config: __ENV.API_CONFIG ? JSON.parse(__ENV.API_CONFIG) : null,
      docker_cpus: __ENV.DOCKER_CPUS || null,
      docker_memory_bytes: __ENV.DOCKER_MEMORY || null,
      git: __ENV.GIT_SHA || null,
      k6_image: __ENV.K6_IMAGE || null,
      finished_at: new Date().toISOString(),
    },
    requests: Object.fromEntries(NAMES.map((n) => [n, {
      served_ms: metric(`${n}_ms`).values,
      any_ms: metric(`${n}_any_ms`).values,
      ok_rate: metric(`${n}_ok`).values.rate,
      shed_rate: metric(`${n}_shed`).values.rate,
      error_rate: metric(`${n}_error`).values.rate,
      cache_hit_rate: metric(`${n}_cache_hit`).values.rate ?? null,
      thresholds: Object.fromEntries(["ok", "ms", "error", "shed", "any_ms"].flatMap((suffix) =>
        Object.entries(metric(`${n}_${suffix}`).thresholds || {}).map(([rule, t]) => [`${n}_${suffix} ${rule}`, t.ok]))),
    }])),
    dropped_iterations: metric("dropped_iterations").values.count || 0,
  };
  const pct = (x) => `${(100 * (x || 0)).toFixed(1)}%`;
  const ms = (x) => (x === undefined ? "-" : `${x.toFixed(1)} ms`);
  const lines = NAMES.map((n) => {
    const r = result.requests[n];
    return `${n.padEnd(10)} ${String(r.any_ms.count).padStart(6)} req  served ${pct(r.ok_rate)}  shed ${pct(r.shed_rate)}` +
      `  errors ${pct(r.error_rate)}  served p50 ${ms(r.served_ms.med)} p95 ${ms(r.served_ms["p(95)"])}` +
      `  any p99 ${ms(r.any_ms["p(99)"])}  cache hits ${pct(r.cache_hit_rate)}`;
  });
  const header = `\n${RUN} (${EXPECT}): ${result.meta.players} players, ${DURATION} at ` +
    `${JSON.stringify(RATES)} req/s, dropped iterations ${result.dropped_iterations}\n`;
  return {
    [`/results/${RUN}.json`]: JSON.stringify(result, null, 2) + "\n",
    stdout: header + lines.join("\n") + "\n",
  };
}
