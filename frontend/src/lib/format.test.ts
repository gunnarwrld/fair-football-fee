import { describe, expect, it } from "vitest";

import { formatDate, formatEur, formatFactor, formatPct, formatValue } from "./format";

describe("formatEur", () => {
  it("compacts millions and thousands", () => {
    expect(formatEur(22_366_822)).toBe("€22.4M");
    expect(formatEur(22_000_000)).toBe("€22M");
    expect(formatEur(950_000)).toBe("€950K");
    expect(formatEur(1_250_000_000)).toBe("€1.3B");
    expect(formatEur(640)).toBe("€640");
  });

  it("moves up a unit when rounding crosses it", () => {
    expect(formatEur(999_960)).toBe("€1M");
  });
});

describe("formatPct", () => {
  it("signs effects with a plus or a true minus", () => {
    expect(formatPct(0.706)).toBe("+70.6%");
    expect(formatPct(-0.312)).toBe("−31.2%");
  });

  it("does not sign a rounded zero", () => {
    expect(formatPct(0.00001)).toBe("0.0%");
    expect(formatPct(-0.00001)).toBe("0.0%");
  });
});

describe("formatFactor", () => {
  it("shows a log half-width as a multiplicative range", () => {
    expect(formatFactor(Math.log(1.61))).toBe("×/÷ 1.61");
  });
});

describe("formatDate", () => {
  it("formats plain calendar dates without a timezone shift", () => {
    expect(formatDate("2026-06-03")).toBe("3 Jun 2026");
  });
});

describe("formatValue", () => {
  it("keeps whole numbers whole and trims decimals", () => {
    expect(formatValue(4370)).toBe("4,370");
    expect(formatValue(22.194)).toBe("22.19");
    expect(formatValue("MID")).toBe("MID");
    expect(formatValue(null)).toBe("—");
  });
});
