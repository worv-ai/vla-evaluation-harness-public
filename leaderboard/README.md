# Paper-reported benchmark results

A focused collection of robot-learning results, especially for useful benchmarks
without a maintained official board. It complements the evaluation harness;
these scores were reported by papers, not reproduced by vla-eval.

Browse one benchmark and comparison at a time. Statistics show collected score
distributions, best scores over paper dates, and reporting-paper counts. They do
not estimate field-wide activity or establish historical SOTA. Official boards
are linked directly rather than duplicated.

## Files

| Path | Responsibility |
| --- | --- |
| `results/<benchmark>.jsonl` | Published rows, one JSON object per line |
| `benchmarks/*.md` | Comparison definitions and machine-readable frontmatter |
| `external.json` | Official boards maintained elsewhere |
| `*.schema.json` | Input contracts |
| `scripts/build.py` | Validate inputs and assemble the static site |
| `site/` | Table, source details, and scoped statistics |
| `maintenance.md` | Internal update priorities |

The migration preserves all 3,576 public-main rows, including their scores,
attribution, dates, and notes. Preservation is not a fresh source audit.
There is no citation crawl, paper census, extraction ledger, or scheduled LLM job.

## Preview

From the repository root:

```bash
uv sync --only-group leaderboard
uv run --no-sync python leaderboard/scripts/build.py
uv run --no-sync python -m http.server 8000 --directory leaderboard/.cache/site
```

Open <http://localhost:8000>. Generated output is ignored by Git.
`build.py --check` validates without writing; `--output PATH` builds elsewhere.

See [CONTRIBUTING.md](CONTRIBUTING.md) for edits and
[update-leaderboard](../skills/update-leaderboard/SKILL.md) for assisted updates.
