"use client";

import { useState } from "react";

import type { Valuation } from "@/lib/api";
import { barFraction, driverRows } from "@/lib/charts";
import { formatEur, formatPct, formatValue } from "@/lib/format";

/**
 * The SHAP decomposition as a table with a diverging bar in each row (so the chart is its own
 * table view). Blue raises the valuation, red lowers it; bar length is the log effect, so a
 * doubling and a halving are the same length.
 */
export function DriverChart({ v }: { v: Valuation }) {
  const rows = driverRows(v);
  const maxAbsPhi = Math.max(...rows.map((r) => Math.abs(r.phi)));
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <section aria-labelledby="drivers-heading" className="space-y-3">
      <div>
        <h2 id="drivers-heading" className="text-lg font-semibold">
          What moves the estimate
        </h2>
        <p className="text-sm text-ink-2">
          Each line multiplies the value of a typical player at today&apos;s league level. Bars use a log scale, so
          +100% and −50% are the same length. These are the model&apos;s associations, not causes.
        </p>
      </div>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="text-left text-ink-2">
            <th scope="col" className="py-1 font-medium">
              Driver
            </th>
            <th scope="col" className="hidden py-1 text-right font-medium whitespace-nowrap sm:table-cell">
              Player&apos;s value
            </th>
            <th scope="col" className="w-2/5 py-1">
              <span className="sr-only">Effect bar</span>
            </th>
            <th scope="col" className="py-1 text-right font-medium">
              Effect
            </th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-y border-border">
            <th scope="row" className="py-2 text-left font-normal">
              Typical player
            </th>
            <td className="hidden sm:table-cell" />
            <td />
            <td className="tabular py-2 text-right">{formatEur(v.baseline_eur)}</td>
          </tr>
          {rows.map((r) => {
            const frac = barFraction(r.phi, maxAbsPhi) * 50;
            const up = r.phi >= 0;
            return (
              <tr
                key={r.key}
                tabIndex={0}
                onPointerEnter={() => setHovered(r.key)}
                onPointerLeave={() => setHovered(null)}
                onFocus={() => setHovered(r.key)}
                onBlur={() => setHovered(null)}
                className={`${r.kind === "adjustment" ? "border-t border-border" : ""} ${hovered === r.key ? "bg-surface" : ""}`}
              >
                <th scope="row" className="py-2 pr-2 text-left font-normal">
                  {r.label}
                  {r.kind === "driver" && (
                    <span className="tabular block text-xs text-ink-2 sm:hidden">{formatValue(r.value)}</span>
                  )}
                  {r.kind === "adjustment" && (
                    <span className="block text-xs text-ink-2">
                      The league-wide shift of Transfermarkt&apos;s latest update round, the same for every player
                    </span>
                  )}
                </th>
                <td className="tabular hidden py-2 pr-3 text-right text-ink-2 sm:table-cell">
                  {r.kind === "driver" ? formatValue(r.value) : ""}
                </td>
                <td className="py-2">
                  <div className="relative h-3" aria-hidden="true">
                    <div className="absolute inset-y-[-4px] left-1/2 w-px bg-axis" />
                    <div
                      className={`absolute inset-y-0 ${up ? "left-1/2 rounded-r bg-up" : "right-1/2 rounded-l bg-down"} ${hovered === r.key ? "opacity-80" : ""}`}
                      style={{ width: `${frac}%` }}
                    />
                    {hovered === r.key && (
                      <div
                        role="tooltip"
                        className="pointer-events-none absolute -top-8 left-1/2 z-10 -translate-x-1/2 rounded-md border border-border bg-surface px-2 py-1 text-xs whitespace-nowrap shadow-sm"
                      >
                        <span className="font-semibold">×{Math.exp(r.phi).toFixed(2)}</span>{" "}
                        <span className="text-ink-2">({formatPct(r.effectPct)}, SHAP {r.phi.toFixed(3)})</span>
                      </div>
                    )}
                  </div>
                </td>
                <td className="tabular py-2 pl-2 text-right">{formatPct(r.effectPct)}</td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="border-t border-border font-semibold">
            <th scope="row" className="py-2 text-left">
              Estimate
            </th>
            <td className="hidden sm:table-cell" />
            <td />
            <td className="tabular py-2 text-right">{formatEur(v.estimate_eur)}</td>
          </tr>
        </tfoot>
      </table>
      <p className="flex gap-5 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="h-2.5 w-4 rounded-r bg-up" /> Raises the valuation
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="h-2.5 w-4 rounded-l bg-down" /> Lowers it
        </span>
      </p>
    </section>
  );
}
