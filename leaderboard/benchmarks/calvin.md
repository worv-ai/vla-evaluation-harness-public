---
benchmark: calvin
display_name: CALVIN
paper_url: https://arxiv.org/abs/2112.03227
metric:
  name: avg_len
  unit: subtasks
  range:
  - 0
  - 5
  higher_is_better: true
suites:
- 1_task
- 2_tasks
- 3_tasks
- 4_tasks
- 5_tasks
---

**Standard**: CALVIN ABC→D split (train A/B/C, evaluate on D) over 1000 evaluation chains; `overall_score` = average number of completed subtasks in chain of 5 (range 0–5).

## Scoring
- `overall_score`: `avg_len` metric (0–5) over 1000 eval chains; `null` if the metric is not `avg_len` or the split is not ABC→D.
- `suite_scores`: optional — some papers report per-chain-length success rates; store under keys like `1_task`, `2_tasks`, ..., `5_tasks` when provided.
- `task_scores`: not used — CALVIN's canonical metric is sequence-level, not task-level.

## Checks
- Is the training split `ABC→D`? `ABCD→D` and `D→D` are different comparisons; do not mix their scores into ABC→D.
- Is the reported metric `avg_len` (0–5)? Rows reporting only success rate percentages without `avg_len` → `null`.
- Does the evaluation use 1000 chains? Any deviation must be recorded in `notes`.

## Methodology axes (record in `notes`, do not null)
- Chain count deviation: note if a paper evaluates on fewer/more than 1000 chains.
- Training data source: CALVIN language-annotated vs full-play subset.
