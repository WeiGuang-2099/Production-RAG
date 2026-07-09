import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DocumentScopePicker } from "./DocumentScopePicker";

const docs = [
  { id: "1", source: "./data/papers/gpt3.pdf", chunks: 178, ingested_at: "" },
  { id: "2", source: "data\\Quiz_ Practice Exam.pdf", chunks: 4, ingested_at: "" },
];

describe("DocumentScopePicker", () => {
  it("renders clean filenames and toggles selection", () => {
    const onChange = vi.fn();
    render(<DocumentScopePicker docs={docs} selected={[]} onChange={onChange} />);

    expect(screen.getByText("gpt3.pdf")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Quiz_ Practice Exam.pdf"));
    expect(onChange).toHaveBeenCalledWith(["data\\Quiz_ Practice Exam.pdf"]);
  });
});
