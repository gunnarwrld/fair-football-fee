"use client";

import Link from "next/link";
import { useId, useState } from "react";

import { ApiError } from "@/lib/api";
import { useDebounced, usePlayerSearch } from "@/lib/hooks";

const POSITION: Record<string, string> = { GK: "Goalkeeper", DEF: "Defender", MID: "Midfielder", FWD: "Forward" };

export function PlayerSearch() {
  const inputId = useId();
  const [query, setQuery] = useState("");
  const debounced = useDebounced(query, 250);
  const { data, error, isLoading } = usePlayerSearch(debounced);
  const results = query.trim().length >= 2 ? (data?.results ?? []) : [];
  const anyUncovered = results.some((r) => !r.has_valuation);

  return (
    <section className="space-y-3" aria-label="Player search">
      <label htmlFor={inputId} className="block text-sm font-medium">
        Search a player
      </label>
      <input
        id={inputId}
        type="search"
        autoComplete="off"
        placeholder="e.g. Saka, Haaland, Rice"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full rounded-md border border-border bg-surface px-3 py-2 text-base outline-none focus-visible:ring-2 focus-visible:ring-link"
      />
      <div aria-live="polite" className="min-h-6 text-sm text-ink-2">
        {error instanceof ApiError
          ? error.message
          : query.trim().length >= 2 && !isLoading && data && results.length === 0
            ? "No players match that name."
            : null}
      </div>
      {results.length > 0 && (
        <ul className={`divide-y divide-border rounded-md border border-border bg-surface transition-opacity ${isLoading ? "opacity-60" : ""}`}>
          {results.map((p) => {
            const meta = [p.current_club, POSITION[p.position_group] ?? p.position_group].filter(Boolean).join(" · ");
            return (
              <li key={p.id}>
                {p.has_valuation ? (
                  <Link href={`/players/${p.id}`} className="flex justify-between gap-4 px-3 py-2 text-ink hover:bg-page hover:no-underline">
                    <span className="font-medium">{p.full_name}</span>
                    <span className="text-sm text-ink-2">{meta}</span>
                  </Link>
                ) : (
                  <div className="flex justify-between gap-4 px-3 py-2 text-ink-2">
                    <span>{p.full_name}</span>
                    <span className="text-sm">Not covered</span>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {anyUncovered && (
        <p className="text-sm text-ink-2">
          &ldquo;Not covered&rdquo;: the model only values Premier League players with at least 450 tracked minutes in
          the last 12 months.
        </p>
      )}
    </section>
  );
}
