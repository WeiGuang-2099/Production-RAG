import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { MessageThread } from "./MessageThread";

test("shows the interpreted-as hint when the condensed question differs", () => {
  render(
    <MessageThread
      messages={[
        { role: "user", content: "what about its cost?" },
        { role: "assistant", content: "It costs...", condensed_question: "What is LoRA's training cost?" },
      ]}
    />,
  );
  expect(screen.getByText(/interpreted as: What is LoRA's training cost\?/)).toBeInTheDocument();
});

test("hides the hint when no condensation happened", () => {
  render(
    <MessageThread
      messages={[
        { role: "user", content: "What is LoRA?" },
        { role: "assistant", content: "LoRA is ..." },
      ]}
    />,
  );
  expect(screen.queryByText(/interpreted as:/)).toBeNull();
});

test("citation markers render as chips and report clicks", async () => {
  const onCitation = vi.fn();
  render(
    <MessageThread
      onCitation={onCitation}
      messages={[
        { role: "user", content: "q" },
        { role: "assistant", content: "relies on attention [1].",
          sources: [{ content: "s", metadata: { citation: 1, source: "a.pdf" } }] },
      ]}
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: "[1]" }));
  expect(onCitation).toHaveBeenCalledWith(1, 1);
});
