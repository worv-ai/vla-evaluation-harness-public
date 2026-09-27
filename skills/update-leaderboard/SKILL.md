---
name: update-leaderboard
description: Add or correct paper-reported benchmark results in vla-eval using targeted, bounded source reviews. Use for leaderboard updates, missing scores, and source audits.
---

# Update leaderboard

Paths below are relative to the vla-eval repository root. If installed elsewhere,
locate the checkout containing `leaderboard/results/` before editing.

1. **Scope.** Read `leaderboard/CONTRIBUTING.md`, `benchmarks/_global.md`, and the
   selected benchmark definitions under `leaderboard/`. Use `maintenance.md` for
   priorities, not eligibility. Follow the user's paper list; otherwise select at
   most five relevant new papers for one benchmark. Prefer missing high-impact
   results and corrections. Do not crawl every citation or refresh every paper.
2. **Read.** Use Paperstack as the canonical paper retrieval interface:
   `paperstack paper metadata arxiv:ID`, then `paperstack paper read arxiv:IDvN
   --outline`, then `--section SECTION --max-chars 16000`. Inspect source tables,
   headers, units, captions, and protocol text together. Use `--documents` for
   supplements and `paperstack paper pdf` for figures or broken source extraction.
   Keep the versioned source URL and table/page location. If Paperstack cannot
   retrieve the source, report the blocked paper; do not silently change sources.
3. **Delegate narrowly.** When subagents are available, use an inexpensive reader
   (Luna, high effort when supported), one paper per task, with only the relevant
   definition and source slices. Ask for method, exact scores/units, source quote,
   location, conditions, and whether baselines were copied. Readers return data;
   only the coordinator edits published files. One targeted retry per unclear
   paper, then leave it unresolved. Do not spawn a second full-paper reviewer by
   default. The coordinator verifies leading scores, contradictions, and a batch
   sample against source cells; escalate only those cases. Without subagents,
   perform the same bounded review directly.
4. **Curate.** Edit `leaderboard/results/<benchmark>.jsonl`. Keep unrelated rows
   and first-added dates. Distinguish re-evaluations from copied baselines; do not
   guess missing cells or average partial results. Preserve evidence and concise
   caveats as described in the contribution guide. A newly supported benchmark
   requires targeted rechecking of relevant existing papers; past review of A/B
   says nothing about whether a paper contains C.
5. **Check.** Run `uv run --group leaderboard python leaderboard/scripts/build.py
   --check`, then the build without `--check`. Inspect affected rows, comparison
   selection, and source details in the preview. Report changed rows, source checks,
   unresolved papers, and measured usage when available. Never infer billed cost
   from model names alone. Stop at the agreed batch; get approval before widening
   scope. Commit or publish only within the user's authorization.

Keep downloads and temporary reader outputs outside tracked files (for example
`leaderboard/.cache/`). No persistent pending queue or duplicate extraction store
is needed. Unresolved work goes in the PR or the user's requested handoff.
