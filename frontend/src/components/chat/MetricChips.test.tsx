import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MetricChips } from "./MetricChips";

test("renders latency, cost, route and guardrail chips", () => {
  render(
    <MetricChips
      message={{
        role: "assistant", content: "a",
        usage: { input_tokens: 50, output_tokens: 10, cost_usd: 0.00213 },
        latency_ms: 845.8, route: "retrieve", attempts: 1,
        guardrails: { pii_redacted: ["email"], flags: [] },
      }}
    />,
  );
  expect(screen.getByText("846 ms")).toBeInTheDocument();
  expect(screen.getByText("$0.00213")).toBeInTheDocument();
  expect(screen.getByText(/route: retrieve · 1/)).toBeInTheDocument();
  expect(screen.getByText(/PII:email/)).toBeInTheDocument();
});

test("renders the interpreted-as chip only when provided", () => {
  const { rerender } = render(
    <MetricChips message={{ role: "assistant", content: "a" }} interpretedAs="What is LoRA?" />,
  );
  expect(screen.getByText(/interpreted as: What is LoRA\?/)).toBeInTheDocument();
  rerender(<MetricChips message={{ role: "assistant", content: "a" }} />);
  expect(screen.queryByText(/interpreted as:/)).toBeNull();
});

test("renders nothing when there is no metadata", () => {
  const { container } = render(<MetricChips message={{ role: "assistant", content: "a" }} />);
  expect(container).toBeEmptyDOMElement();
});
