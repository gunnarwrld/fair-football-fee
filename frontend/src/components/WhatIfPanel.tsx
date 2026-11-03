"use client";

import { useId, useMemo, useState } from "react";

import { ApiError, type InputName, type Inputs, type Scenario, type Valuation } from "@/lib/api";
import { formatEur, formatPct, formatValue } from "@/lib/format";
import { useDebounced, useModel, useWhatIf } from "@/lib/hooks";

const CONTROLS: { name: InputName; label: string; step: number }[] = [
  { name: "age_years", label: "Age", step: 0.5 },
  { name: "minutes_last_12m", label: "Minutes, last 12 months", step: 10 },
  { name: "appearances_last_12m", label: "Appearances, last 12 months", step: 1 },
  { name: "goals_last_12m", label: "Goals, last 12 months", step: 1 },
  { name: "assists_last_12m", label: "Assists, last 12 months", step: 1 },
];

type Values = Partial<Record<InputName, number>>;

function startValues(inputs: Inputs): Values {
  const out: Values = {};
  for (const { name } of CONTROLS) {
    const value = inputs[name];
    if (typeof value === "number") out[name] = value;
  }
  return out;
}

/** Only the inputs that were moved: the API keeps every other count at its per-minute rate. */
function toScenario(values: Values, start: Values): Scenario {
  const scenario: Scenario = {};
  for (const { name } of CONTROLS) {
    const now = values[name];
    if (now !== undefined && start[name] !== undefined && Math.abs(now - (start[name] as number)) > 1e-9) scenario[name] = now;
  }
  return scenario;
}

export function WhatIfPanel({ current }: { current: Valuation }) {
  const baseId = useId();
  const { data: model } = useModel();
  const start = useMemo(() => startValues(current.inputs), [current.inputs]);
  const [values, setValues] = useState<Values>(start);
  const scenario = useMemo(() => toScenario(values, start), [values, start]);
  const debounced = useDebounced(scenario, 300);
  const { data, error, isLoading } = useWhatIf(current.player.id, debounced);
  const active = Object.keys(debounced).length > 0;
  const bounds = model?.whatif_bounds;

  return (
    <section aria-labelledby={`${baseId}-heading`} className="space-y-4 rounded-lg border border-border bg-surface p-4 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 id={`${baseId}-heading`} className="text-lg font-semibold">
            What if?
          </h2>
          <p className="text-sm text-ink-2">
            Move a slider to change the player&apos;s record. Counts you don&apos;t touch keep their per-minute rate, so
            more minutes means the same player playing more.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setValues(start)}
          disabled={Object.keys(scenario).length === 0}
          className="rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-40"
        >
          Reset
        </button>
      </div>

      <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
        {CONTROLS.map(({ name, label, step }) => {
          const id = `${baseId}-${name}`;
          const range = bounds?.[name];
          const value = values[name];
          if (value === undefined || !range) {
            return (
              <p key={name} className="text-sm text-ink-2">
                {label}: {range ? "not available for this player" : "loading…"}
              </p>
            );
          }
          const moved = name in scenario;
          return (
            <div key={name} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1">
              <label htmlFor={id} className="text-sm">
                {label}
              </label>
              <output htmlFor={id} className="tabular text-sm font-semibold">
                {formatValue(value)}
                {moved && <span className="font-normal text-ink-2"> (now {formatValue(start[name])})</span>}
              </output>
              <input
                id={id}
                type="range"
                min={range[0]}
                max={range[1]}
                step={step}
                value={value}
                onChange={(e) => setValues((v) => ({ ...v, [name]: Number(e.target.value) }))}
                className="col-span-2 w-full accent-link"
              />
            </div>
          );
        })}
      </div>

      <div aria-live="polite" className={`border-t border-border pt-4 transition-opacity ${isLoading ? "opacity-60" : ""}`}>
        {!active ? (
          <p className="text-sm text-ink-2">The estimate under your scenario appears here.</p>
        ) : error ? (
          <p role="alert" className="text-sm">
            <span aria-hidden="true">⚠ </span>
            {error instanceof ApiError
              ? error.status === 429
                ? `Too many what-if requests. Try again in ${error.retryAfterSeconds ?? 60} seconds.`
                : error.message
              : "The what-if couldn't be computed."}
          </p>
        ) : data ? (
          <div className="space-y-4">
            <div>
              <p className="text-sm text-ink-2">What-if estimate</p>
              <p className="text-3xl font-semibold">{formatEur(data.estimate_eur)}</p>
              <p className="text-sm">
                <span className="font-semibold">{formatPct(data.change_pct)}</span>
                <span className="text-ink-2">
                  {" "}
                  vs today&apos;s {formatEur(data.current_estimate_eur)} · {Math.round(data.interval.level * 100)}% range{" "}
                  {formatEur(data.interval.low_eur)} – {formatEur(data.interval.high_eur)}
                </span>
              </p>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-ink-2">
                  <th scope="col" className="py-1 font-medium">
                    Input
                  </th>
                  <th scope="col" className="py-1 text-right font-medium">
                    Today
                  </th>
                  <th scope="col" className="py-1 text-right font-medium">
                    What-if
                  </th>
                </tr>
              </thead>
              <tbody>
                {CONTROLS.map(({ name, label }) => {
                  const before = current.inputs[name];
                  const after = data.inputs[name];
                  if (before === after) return null;
                  return (
                    <tr key={name} className="border-t border-border">
                      <th scope="row" className="py-1.5 text-left font-normal">
                        {label}
                        {!(name in debounced) && <span className="block text-xs text-ink-2">kept its per-minute rate</span>}
                      </th>
                      <td className="tabular py-1.5 text-right text-ink-2">{formatValue(before)}</td>
                      <td className="tabular py-1.5 text-right font-semibold">{formatValue(after)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <details className="text-sm">
              <summary className="cursor-pointer text-ink-2">Model features recomputed ({data.changes.length})</summary>
              <ul className="mt-2 space-y-0.5">
                {data.changes.map((c) => (
                  <li key={c.feature} className="tabular text-ink-2">
                    {c.feature.replaceAll("_", " ")}: {formatValue(c.before)} → {formatValue(c.after)}
                  </li>
                ))}
              </ul>
            </details>
          </div>
        ) : null}
      </div>
    </section>
  );
}
