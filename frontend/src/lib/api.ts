/**
 * Typed access to the fair-football-fee API (`/v1`).
 *
 * The types come from the API's OpenAPI schema (`npm run api:types` with the API running), so a
 * backend change that breaks the frontend fails `npm run typecheck` instead of failing in the
 * browser. Requests go straight from the browser to the API: the what-if rate limit is per
 * client IP, and calls from the Next.js server would put every visitor in one bucket.
 */
import type { components } from "./api-types";

type Schemas = components["schemas"];
export type Valuation = Schemas["ValuationOut"];
export type WhatIf = Schemas["WhatIfOut"];
export type ModelInfo = Schemas["ModelOut"];
export type PlayerHit = Schemas["PlayerHitOut"];
export type PlayerSearch = Schemas["PlayerSearchOut"];
export type Driver = Schemas["DriverOut"];
export type Inputs = Schemas["InputsOut"];
export type InputName = keyof Inputs;
export type Scenario = Partial<Record<InputName, number>>;

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/v1";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  const retry = Number(res.headers.get("Retry-After")) || undefined;
  try {
    const detail = (await res.json())?.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail) && "message" in detail) {
      return new ApiError(res.status, String(detail.code), String(detail.message), retry);
    }
    if (Array.isArray(detail) && detail.length > 0) {
      // FastAPI request validation: [{loc, msg, type}, ...]
      const message = detail.map((d: { msg?: string }) => d.msg ?? "invalid input").join("; ");
      return new ApiError(res.status, "validation_error", message, retry);
    }
  } catch {
    // not JSON: fall through to a generic message
  }
  return new ApiError(res.status, "http_error", `Request failed (${res.status})`, retry);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network", `The API can't be reached at ${API_BASE}. Is it running?`);
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

export const api = {
  model: () => request<ModelInfo>("/model"),
  search: (q: string, limit = 10) =>
    request<PlayerSearch>(`/players?${new URLSearchParams({ q, limit: String(limit) })}`),
  valuation: (playerId: number) => request<Valuation>(`/players/${playerId}/valuation`),
  whatIf: (playerId: number, scenario: Scenario) =>
    request<WhatIf>("/valuations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player_id: playerId, scenario }),
    }),
};
