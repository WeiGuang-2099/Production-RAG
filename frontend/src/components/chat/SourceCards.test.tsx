import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SourceCards } from "./SourceCards";

test("renders a citation badge and source name", async () => {
  render(
    <SourceCards sources={[{ content: "snippet text", metadata: { citation: 1, source: "doc.pdf", score: 0.82 } }]} />,
  );
  await userEvent.click(screen.getByRole("button", { name: /sources \(1\)/i }));
  expect(screen.getByText("[1]")).toBeInTheDocument();
  expect(screen.getByText(/doc\.pdf/)).toBeInTheDocument();
});

test("renders nothing for empty sources", () => {
  const { container } = render(<SourceCards sources={[]} />);
  expect(container).toBeEmptyDOMElement();
});
