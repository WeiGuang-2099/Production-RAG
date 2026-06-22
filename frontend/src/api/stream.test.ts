import { streamNdjson } from "./stream";

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

test("parses newline-delimited events, including a split line across chunks", async () => {
  const body = streamFrom([
    '{"event":"sources","sources":[]}\n{"event":"to',
    'ken","token":"Hi"}\n{"event":"done","answer":"Hi"}\n',
  ]);
  const events = [];
  for await (const e of streamNdjson(body)) events.push(e);
  expect(events.map((e) => e.event)).toEqual(["sources", "token", "done"]);
});

test("yields a trailing line with no final newline", async () => {
  const body = streamFrom(['{"event":"token","token":"x"}']);
  const events = [];
  for await (const e of streamNdjson(body)) events.push(e);
  expect(events).toHaveLength(1);
});
