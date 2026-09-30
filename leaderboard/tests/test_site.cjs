const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  paperMonth,
  firstParty,
  score,
  comparisons,
  ranked,
  statistics,
  svgPlot,
} = require("../site/app.js");
const row = (name, value, url = "https://arxiv.org/abs/2501.12345v2") => ({
  display_name: name,
  overall_score: value,
  reported_paper: url,
  date_added: "2026-09-27",
});
test("zero is a score; null and missing are not", () => {
  assert.equal(score(row("A", 0), "overall_score"), 0);
  assert.equal(score(row("A", null), "overall_score"), null);
  assert.equal(score(row("A", 50), "suite_scores.absent"), null);
});
test("rank ties and weight variants stay separate", () => {
  const rows = [row("B", 10), row("A", 10), row("C", 5), row("Missing", null)];
  assert.deepEqual(
    ranked(rows, "overall_score").map((x) => x.rank),
    [1, 1, 3],
  );
  assert.equal(ranked(rows, "overall_score", false)[0].row.display_name, "C");
});
test("paper dates come from arxiv, never insertion dates", () => {
  assert.equal(paperMonth("https://arxiv.org/pdf/2501.12345v2.pdf"), "2025-01");
  assert.equal(paperMonth("https://example.com/2501.12345"), null);
  assert.equal(paperMonth("https://arxiv.org/abs/2513.12345"), null);
});
test("first party requires known matching paper identities", () => {
  assert.equal(
    firstParty({
      reported_paper: "https://arxiv.org/abs/2501.12345v1",
      model_paper: "https://arxiv.org/pdf/2501.12345v2",
    }),
    true,
  );
  assert.equal(firstParty({ reported_paper: "https://example.org" }), false);
});
test("comparison picker never combines independent dimensions or includes reported_avg", () => {
  const bm = {
    aggregation: "forbidden",
    metric: { unit: "%", range: [0, 100] },
    suites: ["vm", "va"],
    tasks: ["reported_avg"],
  };
  const rows = [
    { suite_scores: { vm: 0, va: 50 }, task_scores: { reported_avg: 25 } },
  ];
  assert.deepEqual(
    comparisons(bm, rows).map((x) => x.key),
    ["suite_scores.vm", "suite_scores.va"],
  );
});
test("history, bins and unique reporting papers respect the selected scores", () => {
  const entries = ranked(
    [
      row("A", 50),
      row("B", 100),
      row("C", 0, "https://arxiv.org/abs/2502.12345"),
      row("D", 70, "https://example.org/paper"),
    ],
    "overall_score",
  );
  const stats = statistics(entries, [0, 100]);
  assert.deepEqual(
    stats.history.map((x) => x.value),
    [100, 100],
  );
  assert.equal(stats.years[0].value, 2);
  assert.equal(stats.histogram[0].value, 1);
  assert.equal(stats.histogram[9].value, 1);
  assert.equal(stats.undated, 1);
  assert.deepEqual(
    statistics(entries, [0, 100], false).history.map((x) => x.value),
    [50, 0],
  );
});
test("plots handle one month and escape source labels", () => {
  const svg = svgPlot([{ label: "2025-01", value: 0, detail: "<script>" }], {
    line: true,
    range: [0, 100],
  });
  assert.ok(!svg.includes("NaN"));
  assert.ok(!svg.includes("<script>"));
  assert.ok(svg.includes("View data"));
});

test("metric labels distinguish percentages from success rates", () => {
  const { metricLabel } = require("../site/app.js");
  assert.equal(
    metricLabel(
      { display_name: "VLABench", metric: { unit: "%" } },
      { key: "suite_scores.in_dist_IS", unit: "%" },
    ),
    "Intention score (%)",
  );
  assert.equal(
    metricLabel(
      { display_name: "CALVIN", metric: { unit: "subtasks" } },
      { key: "suite_scores.5_tasks", unit: "%" },
    ),
    "Chain success rate (%)",
  );
});

test("chart drilldowns preserve source identities and bin boundaries", () => {
  const a = row("A", 50),
    b = row("B", 100, "https://arxiv.org/abs/2502.12345"),
    c = row("C", 40, "https://arxiv.org/abs/2503.12345");
  const stats = statistics(ranked([a, b, c], "overall_score"), [0, 100]);
  assert.equal(stats.history[2].rows[0], b);
  assert.deepEqual(stats.histogram[5].rows, [a]);
  assert.deepEqual(stats.histogram[9].rows, [b]);
  assert.equal(stats.years[0].rows.length, 3);
  assert.equal(stats.years[0].value, 3);
});
test("protocol formatting supports headings and lists without executing HTML", () => {
  const { protocolHTML } = require("../site/app.js");
  const result = protocolHTML(
    "## Scoring\n- **Mean** of `a`\n- <img src=x onerror=alert(1)>\n\n[Source](https://example.com)",
  );
  assert.ok(result.includes("<h3>Scoring</h3>"));
  assert.ok(result.includes("<ul>"));
  assert.ok(result.includes("<code>a</code>"));
  assert.ok(!result.includes("<img"));
  assert.ok(result.includes('href="https://example.com"'));
});
test("zero-count bins remain focusable and selectable", () => {
  const plot = svgPlot(
    [
      { label: "0–10", value: 0 },
      { label: "10–20", value: 1 },
    ],
    { unit: "entries" },
  );
  assert.ok(plot.includes('data-point="0"'));
  assert.ok(plot.includes('tabindex="0" role="button"'));
  assert.ok(plot.includes("0 entries"));
});
