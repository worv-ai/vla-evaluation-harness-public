# Updating results

1. Read [shared rules](benchmarks/_global.md) and the relevant benchmark definition.
2. Read the primary source with Paperstack. Add or correct rows in
   `results/<benchmark>.jsonl`; [the schema](results.schema.json) lists fields.
3. Preserve existing row identity and `date_added` on corrections; set `updated`.
   A row is identified by `(benchmark, model, weight_type)`. Distinguish genuinely
   different method variants in `model`, not by silently overwriting them.
4. Validate and inspect the preview, including the changed comparison and source
   details. Describe source checks and unresolved cases in the PR.

```bash
uv sync --only-group leaderboard --only-group dev
uv run --no-sync python leaderboard/scripts/build.py --check
uv run --no-sync pytest leaderboard/tests
node --test leaderboard/tests/test_site.cjs
uv run --no-sync python leaderboard/scripts/build.py
```

New rows should include a versioned `reported_paper`, `reported_table` (or page /
figure), a short literal `evidence` excerpt, the actual `curated_by` identity, and
`score_basis` for aggregates. Use `model_paper` for the method's origin; the
reporting source may be different. Unknown weights remain `unknown`.

Do not fill missing scores with zero. A missing aggregate can coexist with
comparable components. Document computed aggregates in `notes` and include all
components required by the benchmark's declared aggregation rule. Validation
catches structural errors; it does not verify that a paper supports a number.

To add a comparison, update its benchmark Markdown and schema if necessary.
Do not combine different task sets or protocols just to produce an overall rank.
To add an external board, edit `external.json`; it has no local result rows.
