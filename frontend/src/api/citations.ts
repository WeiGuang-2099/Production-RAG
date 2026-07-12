// Rewrites grounded-answer citation markers [n] into markdown links on a
// custom "citation:" protocol, so react-markdown renders them through the
// `a` component override as clickable chips. Only bare numeric markers
// match; [CLS], array[i] etc. pass through. Known edge (accepted in the
// spec): [n] inside code blocks would also be rewritten - this corpus's
// answers do not contain code blocks.
export function linkCitations(text: string): string {
  return text.replace(/\[(\d+)\]/g, (_m, n: string) => `[[${n}]](citation:${n})`);
}
