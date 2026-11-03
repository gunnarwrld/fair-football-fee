#!/usr/bin/env bash
# Load test of the API container: `make loadtest`. Needs `make db` and `make register` first.
#
# 1. (Re)starts the API container with the per-IP what-if limit lifted: k6 is a single client (one IP),
#    and this measures the API's capacity, not the limiter (tests/test_phase5_api.py covers that).
# 2. Runs loadtest/api.js three times, 60 s each:
#      redis_up          the target load, Redis running                 -> everything fast
#      redis_down        the same load, Redis stopped                  -> excess shed with 503s, no 500s
#      redis_down_light  a quarter of that load, Redis still stopped   -> everyone served, just slower
# 3. Restarts Redis and the API with its normal settings, then writes reports/phase7_load_test.md.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
K6_IMAGE=$(docker compose --profile loadtest config --images | grep k6)
export HOST_UID HOST_GID
HOST_UID=$(id -u)
HOST_GID=$(id -g)
mkdir -p reports/loadtest

restore() {
    docker compose start redis >/dev/null
    docker compose --profile app up -d --wait api >/dev/null  # back to the configured limit
}
trap restore EXIT

RATE_LIMIT_PER_MINUTE=1000000 docker compose --profile app up -d --build --wait api || exit 1
# The serving settings as the container sees them, recorded in each run's JSON for the report
api_config=$(docker exec fff-api python -c '
import json, os
from src.config import settings
from src.db.session import engine
print(json.dumps({
    "workers": int(os.environ.get("WEB_CONCURRENCY", "1")),
    "max_concurrent_requests": settings.max_concurrent_requests,
    "db_connections": engine.pool.size() + engine.pool._max_overflow,
    "db_pool_timeout_s": settings.db_pool_timeout_seconds,
}))') || exit 1
meta=(
    -e API_CONFIG="$api_config"
    -e DOCKER_CPUS="$(docker info --format '{{.NCPU}}')"
    -e DOCKER_MEMORY="$(docker info --format '{{.MemTotal}}')"
    -e GIT_SHA="$(git describe --always --dirty)"
    -e K6_IMAGE="$K6_IMAGE"
)
k6() {  # k6 <run name> <expect> <redis state> [extra k6 flags]
    docker compose --profile loadtest run --rm -T k6 run -q "${meta[@]}" -e RUN="$1" -e EXPECT="$2" -e REDIS="$3" \
        "${@:4}" /scripts/api.js
}

failed=0
k6 redis_up fast up || failed=1
docker compose stop redis >/dev/null
k6 redis_down shed down || failed=1
k6 redis_down_light served down -e VALUATION_RATE=50 -e SEARCH_RATE=10 -e WHATIF_RATE=5 || failed=1
trap - EXIT
restore

"$PY" scripts/loadtest_report.py || failed=1
exit $failed
