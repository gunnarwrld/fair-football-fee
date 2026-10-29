"use client";

import Link from "next/link";
import { useEffect } from "react";

import { ApiError, type Valuation } from "@/lib/api";
import { formatDate, formatEur, formatPct, formatValue } from "@/lib/format";
import { useValuation } from "@/lib/hooks";

import { DriverChart } from "./DriverChart";
import { IntervalChart } from "./IntervalChart";
import { WhatIfPanel } from "./WhatIfPanel";

const POSITION: Record<string, string> = { GK: "Goalkeeper", DEF: "Defender", MID: "Midfielder", FWD: "Forward" };

export function PlayerView({ playerId }: { playerId: number }) {
  const valid = Number.isInteger(playerId) && playerId > 0;
  const { data, error } = useValuation(valid ? playerId : null);

  useEffect(() => {
    if (data) document.title = `${data.player.full_name} · Fair Football Fee`;
  }, [data]);

  if (!valid) return <Notice title="Player not found" body="That link doesn't point to a player." />;
  if (error) {
    const body =
      error instanceof ApiError ? error.message : "Something went wrong while loading this valuation. Try again.";
    const title =
      error instanceof ApiError && error.code === "no_valuation" ? "Not covered by the model" : "Valuation unavailable";
    return <Notice title={title} body={body} />;
  }
  if (!data) return <p className="text-ink-2">Loading valuation…</p>;

  return (
    <article className="space-y-12">
      <PlayerHeader v={data} />
      <Headline v={data} />
      <IntervalChart v={data} />
      <DriverChart v={data} />
      <WhatIfPanel key={data.player.id} current={data} />
      <ReadOnlyContext v={data} />
      <p className="text-sm text-ink-2">{data.disclaimer}</p>
    </article>
  );
}

function Notice({ title, body }: { title: string; body: string }) {
  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-semibold">{title}</h1>
      <p className="max-w-2xl text-ink-2">{body}</p>
      <Link href="/">Search another player</Link>
    </div>
  );
}

function PlayerHeader({ v }: { v: Valuation }) {
  const p = v.player;
  const meta = [
    p.current_club,
    POSITION[p.position_group] ?? p.position_group,
    p.age !== null && p.age !== undefined ? `age ${Math.floor(p.age)}` : null,
    p.nationality,
  ].filter(Boolean);
  return (
    <header className="space-y-1">
      <h1 className="text-3xl font-semibold tracking-tight">{p.full_name}</h1>
      <p className="text-ink-2">{meta.join(" · ")}</p>
      <p className="text-sm text-ink-2">Valued as of {formatDate(v.as_of_date)}, using only data from before that day</p>
    </header>
  );
}

function Headline({ v }: { v: Valuation }) {
  const market = v.market_value;
  const inside = market ? market.value_eur >= v.interval.low_eur && market.value_eur <= v.interval.high_eur : null;
  return (
    <section aria-label="Valuation" className="grid gap-8 sm:grid-cols-[auto_1fr] sm:items-end">
      <div>
        <p className="text-sm text-ink-2">Model estimate</p>
        <p className="text-5xl font-semibold tracking-tight">{formatEur(v.estimate_eur)}</p>
        <p className="mt-1 text-sm text-ink-2">
          {Math.round(v.interval.level * 100)}% range {formatEur(v.interval.low_eur)} – {formatEur(v.interval.high_eur)}
        </p>
      </div>
      {market && (
        <dl className="grid grid-cols-2 gap-4 sm:max-w-md">
          <div className="rounded-lg border border-border bg-surface p-3">
            <dt className="text-sm text-ink-2">Transfermarkt value, {formatDate(market.date)}</dt>
            <dd className="text-2xl font-semibold">{formatEur(market.value_eur)}</dd>
          </div>
          <div className="rounded-lg border border-border bg-surface p-3">
            <dt className="text-sm text-ink-2">Model vs Transfermarkt</dt>
            <dd className="text-2xl font-semibold">{formatPct(v.estimate_eur / market.value_eur - 1)}</dd>
            <dd className="text-xs text-ink-2">
              {inside ? "Transfermarkt's value is inside the range" : "Transfermarkt's value is outside the range"}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}

function ReadOnlyContext({ v }: { v: Valuation }) {
  const c = v.context;
  const hasInjuryData = c.injury_days_12m !== null && c.injury_days_12m !== undefined;
  return (
    <section aria-labelledby="context-heading" className="space-y-3">
      <h2 id="context-heading" className="text-lg font-semibold">
        Not adjustable
      </h2>
      <dl className="grid gap-4 text-sm sm:grid-cols-2">
        <div className="space-y-1">
          <dt className="font-medium">Injury history</dt>
          <dd className="text-ink-2">
            {hasInjuryData
              ? `${formatValue(c.injury_days_12m)} days injured in the last 12 months, ${formatValue(c.injury_days_24m)} in the last 24 (${formatValue(c.injury_count_24m)} injuries).`
              : "Not in the injury source, so the model treats it as unknown."}
          </dd>
          <dd className="text-xs text-ink-2">
            Injury days mostly tell the model why a player&apos;s minutes are low, so changing them on their own gives
            misleading answers.
          </dd>
        </div>
        <div className="space-y-1">
          <dt className="font-medium">Contract</dt>
          <dd className="text-ink-2">
            {c.contract_expires_at ? `Runs until ${formatDate(c.contract_expires_at)}.` : "End date unknown."}
          </dd>
          <dd className="text-xs text-ink-2">
            Not used by the model: past contract dates can&apos;t be reconstructed from open data, and a test showed no
            detectable effect.
          </dd>
        </div>
      </dl>
    </section>
  );
}
