import { describe, expect, it } from "vitest";
import { displaySource } from "./sources";

describe("displaySource", () => {
  it("maps the graph sentinel to a readable label", () => {
    expect(displaySource("graph")).toBe("Knowledge graph");
  });
  it("returns the basename of a posix path", () => {
    expect(displaySource("./data/papers/gpt3.pdf")).toBe("gpt3.pdf");
  });
  it("returns the basename of a windows path, keeping spaces", () => {
    expect(displaySource("data\\Quiz_ Practice Exam.pdf")).toBe("Quiz_ Practice Exam.pdf");
  });
  it("shows host and last segment for a URL", () => {
    expect(displaySource("https://example.com/docs/intro.html")).toBe("example.com/intro.html");
  });
  it("passes through an unknown bare value", () => {
    expect(displaySource("unknown")).toBe("unknown");
  });
});
