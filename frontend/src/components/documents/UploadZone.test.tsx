import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { UploadZone } from "./UploadZone";

test("rejects an unsupported file type without calling onFile", async () => {
  const onFile = vi.fn();
  render(<UploadZone onFile={onFile} />);
  const input = screen.getByTestId("file-input") as HTMLInputElement;
  await userEvent.upload(
    input,
    new File(["x"], "evil.exe", { type: "application/octet-stream" }),
    { applyAccept: false },
  );
  expect(onFile).not.toHaveBeenCalled();
  expect(screen.getByText(/unsupported/i)).toBeInTheDocument();
});

test("accepts a markdown file", async () => {
  const onFile = vi.fn();
  render(<UploadZone onFile={onFile} />);
  const input = screen.getByTestId("file-input") as HTMLInputElement;
  await userEvent.upload(input, new File(["# hi"], "doc.md", { type: "text/markdown" }));
  expect(onFile).toHaveBeenCalledTimes(1);
});
