# Shared result rules

- Compare the same benchmark version, task set, evaluation setting, and metric.
  Keep incompatible dimensions separate; missing aggregates do not hide valid
  component scores. Do not infer comparable scores from a method name alone.
- Preserve the source value and unit. Convert fractions to percentages explicitly.
  Compute an aggregate only from every required component under a declared rule;
  record the calculation. A rounded source aggregate may differ slightly.
- Attribute a measurement to the paper that ran it. A copied baseline is not a
  new experiment. A real re-evaluation may have its own row and reporting source.
- Training budget, checkpoint selection, seeds, and trial counts are context to
  record in notes, not general exclusion gates. Do not guess unknown conditions.
- Prefer useful paper-reported coverage with readable caveats over reproduction
  requirements. Include source location and evidence for new or corrected values.
  Unresolved source ambiguity belongs in the PR, not in an invented score.

Benchmark definitions specify comparisons. Do not silently reinterpret inherited
rows when changing a definition; inspect affected sources and show the data diff.
