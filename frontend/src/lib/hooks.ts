"use client";

/**
 * Data hooks over the API client. SWR deduplicates requests shared by several components (the
 * model info feeds the footer, the about page and the what-if bounds) and keeps the previous
 * result on screen while a new one loads, so search and what-if never flash empty.
 */
import { useEffect, useState } from "react";
import useSWR, { type SWRConfiguration } from "swr";

import { ApiError, api, type InputName, type Scenario } from "./api";

// A 4xx answer won't change on retry (unknown player, invalid scenario, rate limit)
const noClientErrorRetry: SWRConfiguration = {
  revalidateOnFocus: false,
  onErrorRetry: (error, _key, _config, revalidate, { retryCount }) => {
    if (error instanceof ApiError && error.status >= 400 && error.status < 500) return;
    if (retryCount >= 2) return;
    setTimeout(() => revalidate({ retryCount }), 2000 * (retryCount + 1));
  },
};

export function useModel() {
  return useSWR("model", api.model, { ...noClientErrorRetry, dedupingInterval: 60_000 });
}

export function useValuation(playerId: number | null) {
  return useSWR(playerId ? ["valuation", playerId] : null, () => api.valuation(playerId as number), noClientErrorRetry);
}

export function usePlayerSearch(query: string) {
  const q = query.trim();
  return useSWR(q.length >= 2 ? ["search", q] : null, () => api.search(q), { ...noClientErrorRetry, keepPreviousData: true });
}

/** A stable key for a scenario: the same inputs in any order hit the same cache entry. */
export function scenarioKey(scenario: Scenario): string {
  const names = Object.keys(scenario).sort() as InputName[];
  return JSON.stringify(names.map((name) => [name, scenario[name]]));
}

export function useWhatIf(playerId: number, scenario: Scenario) {
  const active = Object.keys(scenario).length > 0;
  return useSWR(active ? ["whatif", playerId, scenarioKey(scenario)] : null, () => api.whatIf(playerId, scenario), {
    ...noClientErrorRetry,
    keepPreviousData: true,
  });
}

/** `value`, updated only after it has stopped changing for `delayMs`. */
export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}
