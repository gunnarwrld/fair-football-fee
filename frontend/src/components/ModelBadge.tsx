"use client";

import Link from "next/link";

import { useModel } from "@/lib/hooks";

/** The active model and data version, read from the API (never hardcoded in the frontend). */
export function ModelBadge() {
  const { data, error } = useModel();
  if (error) return <span>Model info unavailable</span>;
  if (!data) return <span aria-hidden="true">&nbsp;</span>;
  return (
    <Link href="/about" className="text-ink-2">
      Model {data.model_version} · data {data.serving_data_version ?? "unknown"}
    </Link>
  );
}
