"""Publication checks independent of source-paper verification."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("leaderboard_build", Path(__file__).parents[1] / "scripts/build.py")
assert SPEC and SPEC.loader
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "site").mkdir()
    (tmp_path / "site/index.html").write_text("<!doctype html><title>Test</title>")
    for name in ("benchmarks.schema.json", "results.schema.json"):
        (tmp_path / name).write_text((build.ROOT / name).read_text())
    (tmp_path / "external.json").write_text("{}")
    (tmp_path / "benchmarks/test.md").write_text(
        "---\nbenchmark: test\ndisplay_name: Test\n"
        "metric: {name: success_rate, unit: '%', range: [0, 100], higher_is_better: true}\n"
        "suites: [a, b]\naggregation: {container: suite_scores, keys: [a, b]}\n---\nDefinition.\n"
    )
    row = {
        "benchmark": "test",
        "model": "example",
        "display_name": "Example",
        "weight_type": "finetuned",
        "curated_by": "reviewer",
        "date_added": "2026-01-01",
        "overall_score": 50,
        "suite_scores": {"a": 40, "b": 60},
        "reported_paper": "https://arxiv.org/abs/2601.12345v2",
    }
    write_rows(tmp_path, [row])
    return tmp_path, row


def write_rows(root, rows):
    (root / "results/test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_preserves_rows(corpus):
    root, row = corpus
    output = root / ".cache/site"
    assert build.build(output, root) == 1
    payload = json.loads((output / "leaderboard.json").read_text())
    assert payload["results"] == [row]
    assert payload["last_updated"] == "2026-01-01"
    assert (output / "protocols/test.md").exists()
    assert "tier" not in payload["benchmarks"]["test"]


@pytest.mark.parametrize(
    "patch,message",
    [
        ({"overall_score": 101}, "outside"),
        ({"overall_score": 75}, "disagrees"),
        ({"overall_score": None, "suite_scores": {}}, "no score"),
        ({"suite_scores": {"a": -1}}, "outside"),
        ({"suite_scores": {"a": 50}, "score_basis": "computed", "notes": "Mean"}, "every component"),
        ({"updated": "2025-01-01"}, "precedes"),
        ({"benchmark": "other"}, "filename"),
        ({"date_added": "2026-02-31"}, "date"),
    ],
)
def test_invalid_rows_fail_before_publication(corpus, patch, message):
    root, row = corpus
    row.update(patch)
    write_rows(root, [row])
    output = root / ".cache/site"
    with pytest.raises(ValueError, match=message):
        build.build(output, root)
    assert not output.exists()


def test_variants_survive_but_duplicate_identity_fails(corpus):
    root, row = corpus
    write_rows(root, [row, dict(row, weight_type="shared")])
    assert len(build.load(root)[1]) == 2
    write_rows(root, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        build.load(root)


def test_partial_results_keep_components(corpus):
    root, row = corpus
    row.update(overall_score=None, suite_scores={"a": 0})
    write_rows(root, [row])
    assert build.load(root)[1][0]["suite_scores"] == {"a": 0}


def test_external_board_rejects_results(corpus):
    root, row = corpus
    config = build.registry(root)["test"]
    config.update(external_only=True, official_leaderboard="https://example.com/board")
    (root / "benchmarks/test.md").unlink()
    (root / "external.json").write_text(json.dumps({"test": config}))
    with pytest.raises(ValueError, match="external-only"):
        build.load(root)


def test_build_does_not_overwrite_sources(corpus):
    root, _ = corpus
    with pytest.raises(ValueError, match="Output must not contain"):
        build.build(root, root)
    with pytest.raises(ValueError, match="outside leaderboard"):
        build.build(root / "site", root)


def test_committed_inputs_validate():
    benchmarks, rows = build.load()
    assert rows
    assert not any("tier" in bm for bm in benchmarks.values())
    assert all(not benchmarks[row["benchmark"]].get("external_only") for row in rows)
