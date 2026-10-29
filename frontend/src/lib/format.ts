/** Number and date formatting shared by every view. Pure functions, unit-tested. */

const dateFormat = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });

const MINUS = "−";
const UNITS = [
  { size: 1e9, suffix: "B" },
  { size: 1e6, suffix: "M" },
  { size: 1e3, suffix: "K" },
];

/**
 * €22.4M, €950K, €1M. Written by hand because Intl's compact notation depends on each runtime's
 * locale data (Node prints "€22.4M", Chrome "€22.4m"), which made the same value render differently.
 */
export function formatEur(value: number): string {
  const sign = value < 0 ? MINUS : "";
  const abs = Math.abs(value);
  const trim = (s: string) => s.replace(/\.0$/, "");
  for (let i = 0; i < UNITS.length; i++) {
    const { size, suffix } = UNITS[i];
    if (abs < size) continue;
    const scaled = (abs / size).toFixed(1);
    if (Number(scaled) >= 1000 && i > 0) {
      const up = UNITS[i - 1]; // rounding crossed into the next unit: 999,960 is €1M, not €1000K
      return `${sign}€${trim((abs / up.size).toFixed(1))}${up.suffix}`;
    }
    return `${sign}€${trim(scaled)}${suffix}`;
  }
  return `${sign}€${Math.round(abs)}`;
}

/** A fraction as a signed percentage: 0.706 → "+70.6%", −0.312 → "−31.2%". */
export function formatPct(fraction: number, digits = 1): string {
  const rounded = Number((fraction * 100).toFixed(digits));
  if (rounded === 0) return `${(0).toFixed(digits)}%`;
  return `${rounded > 0 ? "+" : MINUS}${Math.abs(rounded).toFixed(digits)}%`;
}

/** An interval half-width on the log scale as a multiplicative factor: 0.477 → "×/÷ 1.61". */
export function formatFactor(logHalfWidth: number): string {
  return `×/÷ ${Math.exp(logHalfWidth).toFixed(2)}`;
}

/** "2026-06-03" → "3 Jun 2026". Dates from the API are plain calendar dates. */
export function formatDate(isoDate: string): string {
  return dateFormat.format(new Date(`${isoDate}T00:00:00Z`));
}

/** A feature value for display: whole numbers stay whole, others get up to two decimals. */
export function formatValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  return Number.isInteger(value) ? value.toLocaleString("en-GB") : value.toLocaleString("en-GB", { maximumFractionDigits: 2 });
}
