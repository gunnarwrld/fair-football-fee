import { Suspense } from "react";

import { PlayerView } from "@/components/PlayerView";

/** With Cache Components, route params are request-time data, so they're read inside Suspense. */
export default function PlayerPage({ params }: PageProps<"/players/[id]">) {
  return (
    <Suspense fallback={<p className="text-ink-2">Loading player…</p>}>
      {params.then(({ id }) => (
        <PlayerView playerId={Number(id)} />
      ))}
    </Suspense>
  );
}
