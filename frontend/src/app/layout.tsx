import type { Metadata } from "next";
import Link from "next/link";

import { ModelBadge } from "@/components/ModelBadge";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Fair Football Fee", template: "%s · Fair Football Fee" },
  description:
    "What a Premier League player's recent record implies about their Transfermarkt crowd valuation, with an honest range.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <header className="border-b border-border">
          <nav className="mx-auto flex w-full max-w-4xl items-center justify-between gap-4 px-4 py-3">
            <Link href="/" className="font-semibold text-ink hover:no-underline">
              Fair Football Fee
            </Link>
            <div className="flex gap-5 text-sm">
              <Link href="/">Search</Link>
              <Link href="/about">About the model</Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-8">{children}</main>
        <footer className="border-t border-border">
          <div className="mx-auto flex w-full max-w-4xl flex-wrap items-center justify-between gap-2 px-4 py-4 text-sm text-ink-2">
            <p>Estimates of the Transfermarkt crowd valuation, not transfer fees.</p>
            <ModelBadge />
          </div>
        </footer>
      </body>
    </html>
  );
}
