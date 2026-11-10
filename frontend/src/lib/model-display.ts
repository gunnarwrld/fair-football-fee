export interface CoverageSummary {
  coverage: number;
  ci: { ci_low: number; ci_high: number };
  median_factor: number;
}

export function displayCoverageForLevel(
  coverage: Record<string, CoverageSummary> | undefined,
  displayLevel: number,
): { level: string; coverage: CoverageSummary } | undefined {
  const level = String(displayLevel);
  const selected = coverage?.[level];
  return selected ? { level, coverage: selected } : undefined;
}

export function modelBoundsErrorMessage(error: unknown): string {
  const reason = error instanceof Error && error.message ? ` ${error.message}` : "";
  return `What-if controls are unavailable because model bounds could not be loaded.${reason}`;
}
