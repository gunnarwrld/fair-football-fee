"use client";

import { formatDate, formatPct } from "@/lib/format";
import { useModel } from "@/lib/hooks";

// Shapes of the free-form `test_metrics` and `recentering` objects in GET /v1/model
interface PointMetrics {
  n: number;
  mdape: number;
  within_25pct: number;
}
interface Coverage {
  coverage: number;
  ci: { ci_low: number; ci_high: number };
  median_factor: number;
}
interface SourceBatch {
  start: string;
  end: string;
  n_rows: number;
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

export function ModelDetails() {
  const { data: m, error } = useModel();
  if (error) return <p className="text-ink-2">Model details are unavailable: {error.message}</p>;
  if (!m) return <p className="text-ink-2">Loading model details…</p>;

  const point = m.test_metrics.point as PointMetrics | undefined;
  const unadjusted = m.test_metrics.point_without_recentering as PointMetrics | undefined;
  const coverage = m.test_metrics.coverage as Record<string, Coverage> | undefined;
  const offset = Number(m.recentering.offset ?? 0);
  const batch = m.recentering.source_batch as SourceBatch | null | undefined;

  return (
    <section aria-labelledby="model-heading" className="space-y-4">
      <h2 id="model-heading" className="text-lg font-semibold">
        The model being served
      </h2>
      <dl className="grid gap-3 sm:grid-cols-3">
        <Tile label="Model" value={m.model_version} detail={`XGBoost, trained ${formatDate(m.trained_at.slice(0, 10))}`} />
        {point && (
          <Tile
            label="Typical error on unseen season"
            value={pct(point.mdape)}
            detail={`median absolute % error on ${point.n.toLocaleString("en-GB")} valuations${unadjusted ? ` (${pct(unadjusted.mdape)} without the update-round adjustment)` : ""}`}
          />
        )}
        {coverage &&
          Object.entries(coverage)
            .slice(0, 1)
            .map(([level, c]) => (
              <Tile
                key={level}
                label={`${Math.round(Number(level) * 100)}% range covers`}
                value={pct(c.coverage)}
                detail={`of real values (95% CI ${pct(c.ci.ci_low)}–${pct(c.ci.ci_high)}); typical width ×/÷ ${c.median_factor.toFixed(2)}`}
              />
            ))}
      </dl>
      <p className="text-sm text-ink-2">
        Interval method: {m.interval_method}. The latest Transfermarkt update round adjusts every estimate by{" "}
        {formatPct(Math.expm1(offset))}
        {batch ? ` (median error on the ${formatDate(batch.start)} round, ${batch.n_rows} valuations)` : ""}. Data version{" "}
        {m.serving_data_version ?? "unknown"}.
      </p>
      <details className="text-sm">
        <summary className="cursor-pointer text-ink-2">What-if ranges (the values the model was trained on)</summary>
        <ul className="mt-2 space-y-0.5 text-ink-2">
          {Object.entries(m.whatif_bounds).map(([name, [lo, hi]]) => (
            <li key={name} className="tabular">
              {name.replaceAll("_", " ")}: {lo.toLocaleString("en-GB")} – {hi.toLocaleString("en-GB")}
            </li>
          ))}
        </ul>
      </details>
      {m.disclaimer && <p className="text-xs text-ink-2">{m.disclaimer}</p>}
    </section>
  );
}

function Tile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <dt className="text-sm text-ink-2">{label}</dt>
      <dd className="text-2xl font-semibold">{value}</dd>
      <dd className="text-xs text-ink-2">{detail}</dd>
    </div>
  );
}
