import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "10-K RAG Studio",
  description: "Chat with the latest 10-K filings of Alphabet, Amazon, Microsoft, Apple, and Tesla.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
