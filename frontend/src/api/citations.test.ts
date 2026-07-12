import { expect, test } from "vitest";
import { linkCitations } from "./citations";

test("rewrites [n] markers into citation links", () => {
  expect(linkCitations("relies on attention [1]. Multi-head [2].")).toBe(
    "relies on attention [[1]](citation:1). Multi-head [[2]](citation:2).",
  );
});

test("leaves text without markers untouched", () => {
  expect(linkCitations("no citations here")).toBe("no citations here");
});

test("ignores non-numeric brackets", () => {
  expect(linkCitations("array[i] and [CLS] token")).toBe("array[i] and [CLS] token");
});
