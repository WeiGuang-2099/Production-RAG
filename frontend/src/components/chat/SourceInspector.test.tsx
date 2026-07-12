import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import type { ChatMessage } from "../../hooks/useChat";
import { SourceInspector } from "./SourceInspector";

const messages: ChatMessage[] = [
  { role: "user", content: "q1" },
  {
    role: "assistant", content: "a1 [1]",
    sources: [{ content: "first snippet", metadata: { citation: 1, source: "a.pdf" } }],
  },
  { role: "user", content: "q2" },
  {
    role: "assistant", content: "a2 [1]",
    sources: [
      { content: "second snippet", metadata: { citation: 1, source: "b.pdf", score: 0.9 } },
      { content: "graph snippet", metadata: { citation: 2, source: "graph" } },
    ],
  },
];

test("defaults to the latest assistant message's sources", () => {
  render(<SourceInspector messages={messages} focused={null} />);
  expect(screen.getByText(/b\.pdf/)).toBeInTheDocument();
  expect(screen.getByText("Knowledge graph")).toBeInTheDocument();
  expect(screen.queryByText(/a\.pdf/)).toBeNull();
});

test("follows the focused message and marks the focused citation", () => {
  render(<SourceInspector messages={messages} focused={{ messageIndex: 1, citation: 1 }} />);
  expect(screen.getByText(/a\.pdf/)).toBeInTheDocument();
  expect(screen.getByTestId("source-card-1")).toHaveAttribute("data-focused", "true");
});

test("renders an idle hint when no message has sources", () => {
  render(<SourceInspector messages={[{ role: "user", content: "q" }]} focused={null} />);
  expect(screen.getByText(/Sources appear here/i)).toBeInTheDocument();
});
