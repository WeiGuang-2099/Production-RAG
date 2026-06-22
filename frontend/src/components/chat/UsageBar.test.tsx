import { render, screen } from "@testing-library/react";
import { UsageBar } from "./UsageBar";

test("renders tokens, cost, latency", () => {
  render(<UsageBar usage={{ input_tokens: 50, output_tokens: 10, cost_usd: 0.0002 }} latency_ms={123} />);
  expect(screen.getByText(/50/)).toBeInTheDocument();
  expect(screen.getByText(/\$0.00020/)).toBeInTheDocument();
  expect(screen.getByText(/123 ms/)).toBeInTheDocument();
});
