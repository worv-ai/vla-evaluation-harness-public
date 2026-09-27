"use strict";

const LABELS = {
  google_robot_vm: "Google Robot · Visual Matching",
  google_robot_va: "Google Robot · Variant Aggregation",
  widowx_vm: "WidowX · Visual Matching",
  libero_spatial: "LIBERO-Spatial",
  libero_object: "LIBERO-Object",
  libero_goal: "LIBERO-Goal",
  libero_90: "LIBERO-90",
  libero_10: "LIBERO-10",
};
const label = (key) =>
  LABELS[key] ||
  key
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\b(Vm|Va|Is|Ps)\b/g, (c) => c.toUpperCase());
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const safeURL = (value) => (/^https?:\/\//i.test(value || "") ? value : null);
function paperID(url) {
  return (
    String(url || "").match(
      /arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$/i,
    )?.[1] || null
  );
}
function paperMonth(url) {
  const id = paperID(url);
  if (!id || +id.slice(2, 4) < 1 || +id.slice(2, 4) > 12) return null;
  return `20${id.slice(0, 2)}-${id.slice(2, 4)}`;
}
function sourceID(url) {
  return paperID(url) || String(url || "").replace(/\/$/, "");
}
function firstParty(row) {
  return Boolean(
    row.reported_paper &&
      row.model_paper &&
      sourceID(row.reported_paper) === sourceID(row.model_paper),
  );
}
function score(row, key) {
  const [container, component] = key.split(".");
  const value = component ? row[container]?.[component] : row[container];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function comparisons(bm, rows) {
  const columns = [];
  if (bm.aggregation !== "forbidden")
    columns.push({
      key: "overall_score",
      name: bm.avg_label || "Overall",
      unit: bm.metric.unit,
      range: bm.metric.range,
    });
  for (const [field, container] of [
    ["suites", "suite_scores"],
    ["tasks", "task_scores"],
  ]) {
    for (const key of bm[field] || []) {
      if (
        key !== "reported_avg" &&
        rows.some((row) => score(row, `${container}.${key}`) !== null)
      )
        columns.push({
          key: `${container}.${key}`,
          name: label(key),
          unit: "%",
          range: [0, 100],
        });
    }
  }
  return columns;
}
function metricLabel(bm, column) {
  if (column?.unit === "%" && bm.metric.unit !== "%")
    return "Chain success rate (%)";
  if (bm.display_name === "VLABench") {
    const key = column?.key || "";
    if (key === "overall_score" || /_PS$|progress_score/.test(key))
      return "Progress score (%)";
    if (/_IS$|intention_score/.test(key)) return "Intention score (%)";
    return "Reported score (%)";
  }
  return `${label(bm.metric.name)} (${column?.unit || bm.metric.unit})`;
}
function ranked(rows, key, higher = true) {
  const result = rows
    .filter((r) => score(r, key) !== null)
    .sort(
      (a, b) =>
        (score(b, key) - score(a, key)) * (higher ? 1 : -1) ||
        a.display_name.localeCompare(b.display_name),
    );
  let rank = 0,
    previous;
  return result.map((row, i) => {
    const value = score(row, key);
    if (value !== previous) rank = i + 1;
    previous = value;
    return { row, value, rank };
  });
}
function statistics(entries, range, higher = true) {
  const byMonth = new Map(),
    years = new Map();
  let undated = 0;
  for (const { row, value } of entries) {
    const month = paperMonth(row.reported_paper);
    if (!month) {
      undated++;
      continue;
    }
    if (
      !byMonth.has(month) ||
      (higher
        ? value > byMonth.get(month).value
        : value < byMonth.get(month).value)
    )
      byMonth.set(month, { value, name: row.display_name });
    const year = month.slice(0, 4);
    if (!years.has(year)) years.set(year, new Set());
    years.get(year).add(sourceID(row.reported_paper));
  }
  let best = higher ? -Infinity : Infinity,
    method = "";
  const history = [...byMonth]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, point]) => {
      if (higher ? point.value > best : point.value < best) {
        best = point.value;
        method = point.name;
      }
      return { label: month, value: best, detail: method };
    });
  const [lo, hi] = range,
    width = (hi - lo) / 10;
  const histogram = Array.from({ length: 10 }, (_, i) => ({
    label: `${+(lo + i * width).toFixed(2)}–${+(lo + (i + 1) * width).toFixed(2)}`,
    value: 0,
  }));
  for (const { value } of entries)
    if (value >= lo && value <= hi)
      histogram[Math.min(9, Math.floor((value - lo) / width))].value++;
  return {
    history,
    histogram,
    years: [...years]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([year, papers]) => ({ label: year, value: papers.size })),
    undated,
  };
}
function format(value) {
  if (value === null) return "—";
  return Number.isInteger(value)
    ? String(value)
    : value.toFixed(2).replace(/0$/, "");
}
function svgPlot(points, { line = false, range, unit = "", width = 680 } = {}) {
  if (!points.length)
    return '<p class="empty">No dated reports in this selection.</p>';
  const W = width,
    H = 240,
    left = 45,
    right = 18,
    top = 16,
    bottom = 40;
  const min = range?.[0] ?? 0,
    max = range?.[1] ?? Math.max(1, ...points.map((p) => p.value));
  const span = W - left - right,
    height = H - top - bottom;
  const times = line
    ? points.map((p) => Date.parse(`${p.label}-01T00:00:00Z`))
    : [];
  const x = (i) =>
    line
      ? times.length === 1
        ? left + span / 2
        : left + ((times[i] - times[0]) / (times.at(-1) - times[0])) * span
      : left + ((i + 0.5) * span) / points.length;
  const y = (v) => top + ((max - v) / (max - min)) * height;
  const ticks = [
    ...new Set(
      Array.from({ length: 5 }, (_, i) =>
        range ? min + ((max - min) * i) / 4 : Math.round((max * i) / 4),
      ),
    ),
  ];
  const grid = ticks
    .map(
      (v) =>
        `<line class="grid" x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left - 8}" y="${y(v) + 4}" text-anchor="end">${format(v)}</text>`,
    )
    .join("");
  let previousLabel = -Infinity;
  const labels = points
    .map((p, i) => {
      const position = x(i);
      const last = i === points.length - 1;
      if (
        !last &&
        (position - previousLabel < 80 || x(points.length - 1) - position < 80)
      )
        return "";
      previousLabel = position;
      return `<text x="${position}" y="${H - 12}" text-anchor="middle">${escapeHTML(p.label)}</text>`;
    })
    .join("");
  const path = line
    ? `<path class="line" d="${points.map((p, i) => `${i ? "H" : "M"}${x(i)}${i ? "V" : ","}${y(p.value)}`).join(" ")}"/>`
    : "";
  const marks = points
    .map((p, i) => {
      const title = `<title>${escapeHTML(`${p.label}: ${format(p.value)}${unit}${p.detail ? " · " + p.detail : ""}`)}</title>`;
      return line
        ? `<circle class="point" cx="${x(i)}" cy="${y(p.value)}" r="4">${title}</circle>`
        : `<rect class="bar" x="${x(i) - (span / points.length) * 0.35}" y="${y(p.value)}" width="${(span / points.length) * 0.7}" height="${y(min) - y(p.value)}" rx="2">${title}</rect>`;
    })
    .join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${line ? "Score history" : "Distribution"}; exact values available in the data table below">${grid}${labels}${path}${marks}</svg><details><summary>View data</summary><table><thead><tr><th scope="col">Period / interval</th><th scope="col">Value</th></tr></thead><tbody>${points.map((p) => `<tr><td>${escapeHTML(p.label)}</td><td>${format(p.value)}${escapeHTML(unit)}${p.detail ? " · " + escapeHTML(p.detail) : ""}</td></tr>`).join("")}</tbody></table></details>`;
}

async function start() {
  const $ = (id) => document.getElementById(id);
  const response = await fetch("leaderboard.json");
  if (!response.ok)
    throw new Error(`Could not load results (${response.status}).`);
  const data = await response.json();
  let page = 0,
    reverse = false,
    view = "results",
    current = [],
    missing = [],
    showMissing = false,
    column,
    config;
  const PAGE_SIZE = 40;
  const keys = Object.keys(data.benchmarks).sort((a, b) =>
    data.benchmarks[a].display_name.localeCompare(
      data.benchmarks[b].display_name,
    ),
  );
  for (const external of [false, true]) {
    const group = document.createElement("optgroup");
    group.label = external ? "Official leaderboards" : "Collected results";
    for (const key of keys.filter(
      (k) => Boolean(data.benchmarks[k].external_only) === external,
    )) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = data.benchmarks[key].display_name;
      group.append(option);
    }
    $("benchmark").append(group);
  }
  const initial = new URLSearchParams(location.hash.slice(1));
  $("benchmark").value = keys.includes(initial.get("benchmark"))
    ? initial.get("benchmark")
    : "simpler_env";
  function sourceLink(url, name) {
    return safeURL(url)
      ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(name)} ↗</a>`
      : "—";
  }
  function detail(row) {
    $("detail-title").textContent = row.display_name;
    const values = [];
    if (row.overall_score != null)
      values.push(["Overall", row.overall_score, config.metric.unit]);
    for (const field of ["suite_scores", "task_scores"])
      for (const [key, value] of Object.entries(row[field] || {}))
        values.push([
          key === "reported_avg"
            ? "Reported average (outside this comparison)"
            : label(key),
          value,
          key.startsWith("reported_avg") ? config.metric.unit : "%",
        ]);
    $("detail-body").innerHTML =
      `<h3>Reported measurement</h3><p>${sourceLink(row.reported_paper, paperID(row.reported_paper) ? "arXiv:" + paperID(row.reported_paper) : "Source paper")}${row.reported_table ? " · " + escapeHTML(row.reported_table) : ""}</p>${row.name_in_paper ? `<p>Table label: ${escapeHTML(row.name_in_paper)}</p>` : ""}${row.evidence ? `<blockquote>${escapeHTML(row.evidence)}</blockquote>` : ""}${row.notes ? `<h3>Context</h3><p>${escapeHTML(row.notes)}</p>` : ""}<h3>Reported scores</h3><table><tbody>${values.map(([key, value, unit]) => `<tr><td>${escapeHTML(key)}</td><td>${format(value)} ${escapeHTML(unit)}</td></tr>`).join("")}</tbody></table><h3>Method and provenance</h3><p>${sourceLink(row.model_paper, "Method paper")} · ${escapeHTML(row.weight_type)} weights${row.params ? " · " + escapeHTML(row.params) + " parameters" : ""}</p><p>Added ${escapeHTML(row.date_added)}${row.updated ? " · Updated " + escapeHTML(row.updated) : ""}. Reviewer: ${escapeHTML(row.curated_by)}.${row.score_basis ? " Aggregate: " + escapeHTML(row.score_basis) + "." : ""}</p>`;
    $("detail").showModal();
  }
  function renderRows() {
    const displayed = showMissing
      ? missing
      : reverse
        ? [...current].reverse()
        : current;
    $("rows").innerHTML =
      displayed
        .slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
        .map(
          ({ row, value, rank }, i) =>
            `<tr><td>${rank ?? "—"}</td><td><button class="method" data-index="${i}">${escapeHTML(row.display_name)}</button></td><td>${escapeHTML(row.params || "—")}</td><td>${escapeHTML(row.weight_type)}</td><td class="score score-value">${format(value)}</td><td class="source">${sourceLink(row.reported_paper, paperMonth(row.reported_paper) || "Paper")}</td></tr>`,
        )
        .join("") ||
      '<tr><td colspan="6" class="empty">No reported scores match this selection.</td></tr>';
    $("rows")
      .querySelectorAll(".method")
      .forEach(
        (button) =>
          (button.onclick = () =>
            detail(
              displayed[page * PAGE_SIZE + Number(button.dataset.index)].row,
            )),
      );
    $("page").textContent = displayed.length
      ? `${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, displayed.length)} of ${displayed.length}`
      : "0 results";
    $("previous").disabled = page === 0;
    $("next").disabled = (page + 1) * PAGE_SIZE >= displayed.length;
    $("sort-score").textContent =
      `Score ${reverse === config.metric.higher_is_better ? "↑" : "↓"}`;
  }
  function renderCharts() {
    const stats = statistics(
      current,
      column.range,
      config.metric.higher_is_better,
    );
    const availableWidth = $("charts").clientWidth;
    const chart = (title, description, points, options) =>
      `<article class="chart"><h3>${title}</h3><p>${description}</p>${svgPlot(points, { ...options, width: Math.max(240, (options.line || availableWidth < 700 ? availableWidth : (availableWidth - 20) / 2) - 44) })}</article>`;
    $("charts").innerHTML =
      chart(
        "Best collected score over paper dates",
        `Running best by reporting paper’s first arXiv month. Later revisions can contain newer results, so this is not a historical SOTA record. ${stats.undated} entries without an arXiv date omitted.`,
        stats.history,
        {
          line: true,
          range: column.range,
          unit: column.unit === "%" ? "%" : "",
        },
      ) +
      chart(
        "Score distribution",
        `Entries with a ${escapeHTML(column.name)} score. Bins include their lower bound; the last includes the maximum.`,
        stats.histogram,
        {},
      ) +
      chart(
        "Reporting papers by year",
        "Unique reporting papers with a score in this comparison, by first arXiv year. Collection coverage varies; this does not measure research activity.",
        stats.years,
        {},
      );
  }
  function render() {
    page = 0;
    const rows = data.results.filter(
      (r) => r.benchmark === $("benchmark").value,
    );
    const selected = rows.filter(
      (r) =>
        (!$("first-party").checked || firstParty(r)) &&
        r.display_name
          .toLowerCase()
          .includes($("search").value.toLowerCase().trim()),
    );
    current = column
      ? ranked(selected, column.key, config.metric.higher_is_better)
      : [];
    missing = selected
      .filter((r) => !column || score(r, column.key) === null)
      .map((row) => ({ row, value: null, rank: null }));
    $("summary").innerHTML =
      `${current.length} scored entries · ${new Set(current.map((x) => sourceID(x.row.reported_paper)).filter(Boolean)).size} reporting papers` +
      (missing.length && view === "results"
        ? ` · <button id="show-missing" aria-pressed="${showMissing}">${showMissing ? "Back to scored entries" : `${missing.length} without this score`}</button>`
        : "");
    if ($("show-missing"))
      $("show-missing").onclick = () => {
        showMissing = !showMissing;
        render();
      };
    if (!missing.length) showMissing = false;
    $("results-view").hidden = view !== "results";
    $("insights-view").hidden = view !== "insights";
    $("results-tab").setAttribute("aria-pressed", String(view === "results"));
    $("insights-tab").setAttribute("aria-pressed", String(view === "insights"));
    renderRows();
    if (view === "insights" && column) renderCharts();
    history.replaceState(
      null,
      "",
      "#" +
        new URLSearchParams({
          benchmark: $("benchmark").value,
          comparison: $("comparison").value,
          view,
        }),
    );
  }
  function chooseBenchmark(wanted) {
    const key = $("benchmark").value;
    config = data.benchmarks[key];
    const external = Boolean(config.external_only);
    $("board").hidden = external;
    $("external").hidden = !external;
    $("comparison-control").hidden = external;
    $("search-control").hidden = external;
    if (external) {
      $("external").innerHTML =
        `<h2>${escapeHTML(config.display_name)}</h2><p>Results are maintained on the benchmark’s official leaderboard.</p>${sourceLink(config.official_leaderboard, "Open official leaderboard")}`;
      history.replaceState(
        null,
        "",
        "#" + new URLSearchParams({ benchmark: key }),
      );
      return;
    }
    const columns = comparisons(
      config,
      data.results.filter((r) => r.benchmark === key),
    );
    $("comparison").innerHTML = columns
      .map(
        (c) =>
          `<option value="${escapeHTML(c.key)}">${escapeHTML(c.name)}</option>`,
      )
      .join("");
    const populated = columns.find((c) =>
      data.results.some((r) => r.benchmark === key && score(r, c.key) !== null),
    );
    if (populated) $("comparison").value = populated.key;
    if (columns.some((c) => c.key === wanted)) $("comparison").value = wanted;
    function chooseColumn() {
      showMissing = false;
      column = columns.find((c) => c.key === $("comparison").value);
      $("board-title").textContent =
        `${config.display_name} · ${column?.name || "Results"}`;
      $("metric").textContent =
        `${metricLabel(config, column)} · ${config.metric.higher_is_better ? "Higher" : "Lower"} is better`;
      $("protocol").href = `protocols/${key}.md`;
      render();
    }
    $("comparison").onchange = chooseColumn;
    chooseColumn();
  }
  $("benchmark").onchange = () => {
    $("search").value = "";
    reverse = false;
    chooseBenchmark();
  };
  $("search").oninput = () => {
    showMissing = false;
    render();
  };
  $("first-party").onchange = render;
  $("results-tab").onclick = () => {
    view = "results";
    render();
  };
  $("insights-tab").onclick = () => {
    view = "insights";
    render();
  };
  $("previous").onclick = () => {
    page--;
    renderRows();
  };
  $("next").onclick = () => {
    page++;
    renderRows();
  };
  $("sort-score").onclick = () => {
    reverse = !reverse;
    page = 0;
    renderRows();
  };
  $("close-detail").onclick = () => $("detail").close();
  $("detail").onclick = (event) => {
    if (event.target === $("detail")) {
      const r = $("detail").getBoundingClientRect();
      if (
        event.clientX < r.left ||
        event.clientX > r.right ||
        event.clientY < r.top ||
        event.clientY > r.bottom
      )
        $("detail").close();
    }
  };
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (view === "insights" && column && !config.external_only)
        renderCharts();
    }, 120);
  });
  $("updated").textContent = data.last_updated
    ? `Latest recorded update: ${data.last_updated}.`
    : "";
  view = initial.get("view") === "insights" ? "insights" : "results";
  chooseBenchmark(initial.get("comparison"));
}
if (typeof module !== "undefined")
  module.exports = {
    metricLabel,
    paperID,
    paperMonth,
    sourceID,
    firstParty,
    score,
    comparisons,
    ranked,
    statistics,
    svgPlot,
  };
if (typeof document !== "undefined")
  start().catch((error) => {
    const el = document.getElementById("error");
    el.hidden = false;
    el.textContent =
      error.message +
      " Build and serve the site using leaderboard/scripts/build.py.";
  });
