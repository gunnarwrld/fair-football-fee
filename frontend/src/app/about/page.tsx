import type { Metadata } from "next";

import { ModelDetails } from "@/components/ModelDetails";

export const metadata: Metadata = { title: "About the model" };

export default function AboutPage() {
  return (
    <article className="max-w-3xl space-y-10">
      <header className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">About the model</h1>
        <p className="text-ink-2">
          Fair Football Fee estimates a Premier League player&apos;s Transfermarkt market value: the crowd-sourced
          valuation, not a transfer fee. Fees also depend on release clauses, add-ons and negotiating leverage, which
          no open dataset records.
        </p>
      </header>

      <ModelDetails />

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">How it works</h2>
        <ul className="list-disc space-y-2 pl-5 text-ink-2">
          <li>
            <span className="text-ink">Only the past.</span> Every input is computed from data dated before the
            valuation day, the same way for training and for this site, and tests check that nothing later leaks in.
          </li>
          <li>
            <span className="text-ink">Relative to the league.</span> The model predicts a player&apos;s value relative to
            the recent league median, so it doesn&apos;t have to guess market inflation.
          </li>
          <li>
            <span className="text-ink">What it doesn&apos;t see.</span> The player&apos;s previous valuation and contract
            length. It answers &ldquo;what is this record worth&rdquo;, not &ldquo;where will the next valuation
            land&rdquo;.
          </li>
          <li>
            <span className="text-ink">Honest ranges.</span> The ranges are split-conformal intervals calibrated on a
            season the model never trained on, wider for players whose values are harder to predict, and checked once
            on a later season.
          </li>
          <li>
            <span className="text-ink">Drivers.</span> Each estimate is broken down with exact TreeSHAP values into
            multiplicative effects. They describe what the model learned, not what would cause a value to change.
          </li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Known limits</h2>
        <ul className="list-disc space-y-2 pl-5 text-ink-2">
          <li>Players over about 29 are valued too generously: Transfermarkt marks age down faster than output falls.</li>
          <li>Players worth under €5M are the hardest to predict, and their ranges miss most often.</li>
          <li>
            When Transfermarkt revalues the whole league, the model catches up one update round later through its
            league-wide adjustment.
          </li>
        </ul>
      </section>
    </article>
  );
}
