"""Validate curated result files and build the standalone leaderboard site."""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def validate(value, schema, location):
    validator = jsonschema.Draft7Validator(schema, format_checker=jsonschema.FormatChecker())
    errors = list(validator.iter_errors(value))
    if errors:
        error = errors[0]
        raise ValueError(f"{location}: {'.'.join(map(str, error.absolute_path))}: {error.message}")


def registry(root):
    benchmarks = read_json(root / "external.json")
    for key, entry in benchmarks.items():
        if not entry.get("external_only") or not entry.get("official_leaderboard"):
            raise ValueError(f"external.json/{key}: an external leaderboard URL is required")
    for path in sorted((root / "benchmarks").glob("*.md")):
        if path.stem == "_global":
            continue
        parts = path.read_text().split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise ValueError(f"{path}: missing YAML frontmatter")
        entry = yaml.safe_load(parts[1])
        key = entry.pop("benchmark", path.stem)
        if key != path.stem or key in benchmarks:
            raise ValueError(f"{path}: duplicate or mismatched benchmark key {key}")
        if entry.get("external_only"):
            raise ValueError(f"{path}: put external boards in external.json")
        benchmarks[key] = entry
    validate(benchmarks, read_json(root / "benchmarks.schema.json"), "benchmarks")
    return dict(sorted(benchmarks.items()))


def validate_row(row, bm, location):
    if bm.get("external_only"):
        raise ValueError(f"{location}: external-only benchmark cannot have local results")
    lo, hi = bm["metric"]["range"]
    overall = row.get("overall_score")
    if overall is None and not row.get("suite_scores") and not row.get("task_scores"):
        raise ValueError(f"{location}: no score")
    if overall is not None and (not math.isfinite(overall) or not lo <= overall <= hi):
        raise ValueError(f"{location}: overall outside [{lo}, {hi}]")
    for field, declared in (("suite_scores", "suites"), ("task_scores", "tasks")):
        for key, score in row.get(field, {}).items():
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f"{location}: {field}.{key} outside [0, 100]")
            if overall is not None and bm.get(declared) and key not in bm[declared]:
                raise ValueError(f"{location}: unknown {field}.{key}")
            suffixes = bm.get("score_key_suffixes")
            if (
                field == "task_scores"
                and suffixes
                and key != "reported_avg"
                and not any(key.endswith("_" + s) for s in suffixes)
            ):
                raise ValueError(f"{location}: {field}.{key} lacks a protocol suffix")
    rule = bm.get("aggregation")
    if rule == "forbidden" and overall is not None:
        raise ValueError(f"{location}: no cross-dimension aggregate allowed")
    if isinstance(rule, dict) and overall is not None:
        values = row.get(rule["container"], {})
        complete = all(key in values for key in rule["keys"])
        if row.get("score_basis") == "computed" and not complete:
            raise ValueError(f"{location}: computed aggregate needs every component")
        if complete:
            mean = sum(values[key] for key in rule["keys"]) / len(rule["keys"])
            if abs(overall - mean) > 0.25:
                raise ValueError(f"{location}: aggregate disagrees with complete components ({mean:.3f})")
    if row.get("score_basis") == "computed" and (not isinstance(rule, dict) or not row.get("notes")):
        raise ValueError(f"{location}: computed aggregate needs a declared formula and notes")
    if row.get("updated", row["date_added"]) < row["date_added"]:
        raise ValueError(f"{location}: updated precedes date_added")


def load(root=ROOT):
    benchmarks = registry(root)
    schema = read_json(root / "results.schema.json")
    results, seen = [], set()
    for path in sorted((root / "results").glob("*.jsonl")):
        if path.stem not in benchmarks:
            raise ValueError(f"{path}: unknown benchmark")
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            location = f"{path.name}:{number}"
            row = json.loads(line)
            validate(row, schema, location)
            if row["benchmark"] != path.stem:
                raise ValueError(f"{location}: benchmark does not match filename")
            identity = row["benchmark"], row["model"], row["weight_type"]
            if identity in seen:
                raise ValueError(f"{location}: duplicate result {identity}; distinguish method variants")
            seen.add(identity)
            validate_row(row, benchmarks[path.stem], location)
            results.append(row)
    results.sort(key=lambda r: (r["benchmark"], r["model"], r["weight_type"]))
    return benchmarks, results


def build(output, root=ROOT):
    benchmarks, results = load(root)
    output = output.resolve()
    source = root.resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError("Output must not contain the source directory")
    if output.is_relative_to(source) and not output.is_relative_to(source / ".cache"):
        raise ValueError("Use .cache/site or an output directory outside leaderboard/")
    output.mkdir(parents=True, exist_ok=True)
    for path in (root / "site").iterdir():
        if path.is_file():
            shutil.copyfile(path, output / path.name)
    index = output / "index.html"
    html = index.read_text()
    for name in ("app.js", "style.css"):
        asset = output / name
        if asset.exists():
            version = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
            html = html.replace(f'"{name}"', f'"{name}?v={version}"')
    index.write_text(html)
    shutil.copytree(root / "benchmarks", output / "protocols", dirs_exist_ok=True)
    payload = {
        "last_updated": max((r.get("updated", r["date_added"]) for r in results), default=None),
        "benchmarks": benchmarks,
        "results": results,
    }
    (output / "leaderboard.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate authored files without writing")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/site")
    args = parser.parse_args()
    try:
        if args.check:
            _, results = load()
            print(f"OK: {len(results)} results")
        else:
            print(f"Built {build(args.output)} results at {args.output}")
    except (ValueError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
