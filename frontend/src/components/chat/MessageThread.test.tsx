import { render, screen } from "@testing-library/react";
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
