"use client";

import { useState } from "react";

import type { Valuation } from "@/lib/api";
import { niceTicks, position } from "@/lib/charts";
import { formatEur } from "@/lib/format";

interface Tip {
  text: string;
  left: number;
}

/**
 * One value axis: the widest interval as a light band, the displayed interval as a darker band,
 * the estimate as a dot and the Transfermarkt value as a diamond. Values close together would
 * collide as direct labels, so identity comes from the legend (which carries the numbers) and the
 * hover/focus tooltip.
 */
export function IntervalChart({ v }: { v: Valuation }) {
  const [tip, setTip] = useState<Tip | null>(null);
  const inner = v.interval;
  const outer = [...v.intervals].sort((a, b) => b.level - a.level)[0] ?? inner;
  const market = v.market_value?.value_eur;

  const lo = Math.min(outer.low_eur, market ?? Infinity);
  const hi = Math.max(outer.high_eur, market ?? -Infinity);
  const pad = (hi - lo) * 0.08;
  const dMin = Math.max(0, lo - pad);
  const dMax = hi + pad;
  const at = (x: number) => position(x, dMin, dMax);
  const show = (text: string, x: number) => () => setTip({ text, left: Math.min(88, Math.max(12, at(x))) });
  const hide = () => setTip(null);
  const pctLevel = (level: number) => `${Math.round(level * 100)}%`;

  const innerText = `${pctLevel(inner.level)} range ${formatEur(inner.low_eur)} – ${formatEur(inner.high_eur)}`;
  const outerText = `${pctLevel(outer.level)} range ${formatEur(outer.low_eur)} – ${formatEur(outer.high_eur)}`;
  const estimateText = `Model estimate ${formatEur(v.estimate_eur)}`;
  const marketText = market !== undefined ? `Transfermarkt value ${formatEur(market)}` : null;

  return (
    <section aria-labelledby="interval-heading" className="space-y-3">
      <div>
        <h2 id="interval-heading" className="text-lg font-semibold">
          How sure is the estimate?
        </h2>
        <p className="text-sm text-ink-2">
          In the season the model never saw, the real value fell inside the {pctLevel(inner.level)} range for roughly{" "}
          {pctLevel(inner.level)} of players. The range is wider for players the model rates well below the league
          median, because their values are harder to predict.
        </p>
      </div>

      <div className="relative h-20 select-none" onPointerLeave={hide}>
        {tip && (
          <div
            role="tooltip"
            className="pointer-events-none absolute -top-1 z-10 -translate-x-1/2 rounded-md border border-border bg-surface px-2 py-1 text-sm font-semibold whitespace-nowrap shadow-sm"
            style={{ left: `${tip.left}%` }}
          >
            {tip.text}
          </div>
        )}
        {niceTicks(dMin, dMax, 4).map((t) => (
          <div key={t} aria-hidden="true">
            <div className="absolute top-6 h-8 w-px bg-grid" style={{ left: `${at(t)}%` }} />
            <div className="tabular absolute top-[3.75rem] -translate-x-1/2 text-xs text-muted" style={{ left: `${at(t)}%` }}>
              {formatEur(t)}
            </div>
          </div>
        ))}
        <div aria-hidden="true" className="absolute top-14 right-0 left-0 h-px bg-axis" />
        <div
          aria-hidden="true"
          className="absolute top-[2.0625rem] h-2.5 rounded bg-band-90"
          style={{ left: `${at(outer.low_eur)}%`, width: `${at(outer.high_eur) - at(outer.low_eur)}%` }}
          onPointerEnter={show(outerText, (outer.low_eur + outer.high_eur) / 2)}
        />
        <div
          aria-hidden="true"
          className="absolute top-[1.75rem] h-[1.125rem] rounded bg-band-80"
          style={{ left: `${at(inner.low_eur)}%`, width: `${at(inner.high_eur) - at(inner.low_eur)}%` }}
          onPointerEnter={show(innerText, v.estimate_eur)}
        />
        {market !== undefined && marketText && (
          <button
            type="button"
            aria-label={marketText}
            className="absolute top-[2.375rem] size-6 -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${at(market)}%` }}
            onPointerEnter={show(marketText, market)}
            onFocus={show(marketText, market)}
            onBlur={hide}
          >
            <span className="absolute inset-[6px] rotate-45 bg-ink ring-2 ring-surface" />
          </button>
        )}
        <button
          type="button"
          aria-label={estimateText}
          className="absolute top-[2.375rem] size-6 -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${at(v.estimate_eur)}%` }}
          onPointerEnter={show(estimateText, v.estimate_eur)}
          onFocus={show(estimateText, v.estimate_eur)}
          onBlur={hide}
        >
          <span className="absolute inset-[5px] rounded-full bg-estimate ring-2 ring-surface" />
        </button>
      </div>

      <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm text-ink-2">
        <li className="flex items-center gap-2">
          <span aria-hidden="true" className="size-3 rounded-full bg-estimate" />
          {estimateText}
        </li>
        {marketText && (
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="size-2.5 rotate-45 bg-ink" />
            {marketText}
          </li>
        )}
        <li className="flex items-center gap-2">
          <span aria-hidden="true" className="h-3 w-5 rounded bg-band-80" />
          {innerText}
        </li>
        <li className="flex items-center gap-2">
          <span aria-hidden="true" className="h-2 w-5 rounded bg-band-90" />
          {outerText}
        </li>
      </ul>
    </section>
  );
}
