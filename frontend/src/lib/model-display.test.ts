import { describe, expect, it } from "vitest";

import { displayCoverageForLevel, modelBoundsErrorMessage } from "./model-display";

describe("displayCoverageForLevel", () => {
  it("selects the configured display interval instead of the first object entry", () => {
    const coverage = {
      "0.9": { coverage: 0.91, ci: { ci_low: 0.89, ci_high: 0.93 }, median_factor: 1.8 },
      "0.8": { coverage: 0.82, ci: { ci_low: 0.80, ci_high: 0.84 }, median_factor: 1.5 },
    };

    expect(displayCoverageForLevel(coverage, 0.8)).toEqual({ level: "0.8", coverage: coverage["0.8"] });
  });
});

describe("modelBoundsErrorMessage", () => {
  it("turns a failed model load into an explicit controls error", () => {
    expect(modelBoundsErrorMessage(new Error("API unavailable"))).toBe(
      "What-if controls are unavailable because model bounds could not be loaded. API unavailable",
    );
  });
});
