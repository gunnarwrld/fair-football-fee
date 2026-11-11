import { PlayerSearch } from "@/components/PlayerSearch";

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">What does a player&apos;s record say they&apos;re worth?</h1>
        <p className="max-w-2xl text-ink-2">
          A model trained on ten Premier League seasons estimates a player&apos;s Transfermarkt crowd valuation from
          their recent minutes, output, age and fitness. Every estimate comes with a range checked against a season
          the model never saw, the reasons behind it, and a what-if panel.
        </p>
      </section>
      <PlayerSearch />
    </div>
  );
}
