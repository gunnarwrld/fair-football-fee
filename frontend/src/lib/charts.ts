/** Chart math for the driver bars and the interval axis. Pure functions, unit-tested. */
import type { Valuation } from "./api";

export type DriverKind = "driver" | "other" | "adjustment";

export interface DriverRow {
  key: string;
  label: string;
  value: number | string | null;
  phi: number; // log-scale effect: the valuation is multiplied by exp(phi)
  effectPct: number; // exp(phi) - 1
  kind: DriverKind;
}

/** The top drivers, then "all other features", then the update-round adjustment. */
export function driverRows(v: Valuation): DriverRow[] {
  const rows: DriverRow[] = v.drivers.map((d) => ({
    key: d.feature,
    label: d.label,
    value: d.value ?? null,
    phi: d.phi,
    effectPct: d.effect_pct,
    kind: "driver",
  }));
  rows.push({
    key: "__other",
    label: "All other features",
    value: null,
    phi: Math.log1p(v.other_drivers_effect_pct),
    effectPct: v.other_drivers_effect_pct,
    kind: "other",
  });
  rows.push({
    key: "__adjustment",
    label: v.adjustment.label ?? "Latest update-round adjustment",
    value: null,
    phi: v.adjustment.phi,
    effectPct: v.adjustment.effect_pct,
    kind: "adjustment",
  });
  return rows;
}

/**
 * The rows multiply the typical-player baseline back to the estimate:
 * (1 + baseline) · Π(1 + effect) = 1 + estimate. Returns the relative error of that identity.
 */
export function decompositionError(v: Valuation): number {
  const product = driverRows(v).reduce((acc, r) => acc * (1 + r.effectPct), 1 + v.baseline_eur);
  return Math.abs(product / (1 + v.estimate_eur) - 1);
}

/** Bar length as a fraction of the half-width, on a log scale so +100% and −50% match. */
export function barFraction(phi: number, maxAbsPhi: number): number {
  return maxAbsPhi > 0 ? Math.min(1, Math.abs(phi) / maxAbsPhi) : 0;
}

/** Round "nice" tick values (1, 2, 2.5 or 5 × 10^k) covering [min, max] with about `count` ticks. */
export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  const rough = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= rough) ?? 10 * power;
  const ticks: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
    ticks.push(Number(t.toPrecision(12)));
  }
  return ticks;
}

/** Map a value in [domainMin, domainMax] to a percentage position along the axis. */
export function position(value: number, domainMin: number, domainMax: number): number {
  return ((value - domainMin) / (domainMax - domainMin)) * 100;
}
