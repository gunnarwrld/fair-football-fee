import { describe, expect, it } from "vitest";

import type { Valuation } from "./api";
import { barFraction, decompositionError, driverRows, niceTicks, position } from "./charts";

/** A valuation built so its drivers multiply exactly back to the estimate (an obviously fake player). */
function fakeValuation(): Valuation {
  const baseline = 24_470_000;
  const phis = [0.534, -0.374, 0.134, -0.114, -0.096];
  const otherPhi = -0.035;
  const offset = -0.2108;
  const total = phis.reduce((a, b) => a + b, 0) + otherPhi + offset;
  return {
    baseline_eur: baseline,
    estimate_eur: Math.expm1(Math.log1p(baseline) + total),
    drivers: phis.map((phi, i) => ({ feature: `f${i}`, label: `Feature ${i}`, value: i, phi, effect_pct: Math.expm1(phi) })),
    other_drivers_effect_pct: Math.expm1(otherPhi),
    adjustment: { label: "Latest update-round adjustment", phi: offset, effect_pct: Math.expm1(offset), source_batch: null },
  } as unknown as Valuation;
}

describe("driverRows", () => {
  it("lists drivers, then other features, then the adjustment", () => {
    const rows = driverRows(fakeValuation());
    expect(rows.map((r) => r.kind)).toEqual(["driver", "driver", "driver", "driver", "driver", "other", "adjustment"]);
    expect(rows[5].phi).toBeCloseTo(-0.035, 12);
  });

  it("multiplies the baseline back to the estimate", () => {
    expect(decompositionError(fakeValuation())).toBeLessThan(1e-12);
  });
});

describe("barFraction", () => {
  it("draws doubling and halving with the same length", () => {
    expect(barFraction(Math.log(2), 1)).toBeCloseTo(barFraction(Math.log(0.5), 1), 12);
  });

  it("never exceeds the half-width", () => {
    expect(barFraction(5, 1)).toBe(1);
    expect(barFraction(0.3, 0)).toBe(0);
  });
});

describe("niceTicks", () => {
  it("covers the range with round, evenly spaced values", () => {
    const ticks = niceTicks(12.4e6, 41.9e6, 4);
    expect(ticks).toEqual([20e6, 30e6, 40e6]);
    expect(niceTicks(0, 1, 4)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });
});

describe("position", () => {
  it("maps the domain onto 0-100%", () => {
    expect(position(15, 10, 20)).toBe(50);
  });
});
