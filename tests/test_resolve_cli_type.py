"""``_resolve_cli_type`` covers nullable arg handling: yaml-null ("None"/"null"/"")
round-tripped through cmd_serve must reach the inner __init__ as Python ``None``
instead of failing in ``int("None")``-style coercion."""

from __future__ import annotations

import pytest

from vla_eval.model_servers.serve import _resolve_cli_type


class TestNullableWrapping:
    def test_optional_int_wraps_none_tokens(self):
        type_fn, is_bool, skip = _resolve_cli_type(int | None, default=None)
        assert not is_bool
        assert not skip
        assert type_fn is not None
        assert type_fn("None") is None
        assert type_fn("null") is None
        assert type_fn("") is None
        assert type_fn("4096") == 4096

    def test_optional_str_wraps_none_tokens(self):
        type_fn, is_bool, skip = _resolve_cli_type(str | None, default=None)
        assert not is_bool
        assert not skip
        assert type_fn is not None
        assert type_fn("None") is None
        assert type_fn("max-autotune") == "max-autotune"

    def test_optional_float_wraps_none_tokens(self):
        type_fn, _, _ = _resolve_cli_type(float | None, default=None)
        assert type_fn is not None
        assert type_fn("None") is None
        assert type_fn("1.5") == 1.5

    def test_plain_int_does_not_swallow_none_string(self):
        """Non-nullable arg: 'None' is a real (and invalid) int — must raise."""
        type_fn, _, _ = _resolve_cli_type(int, default=0)
        assert type_fn is int
        with pytest.raises(ValueError):
            type_fn("None")

    def test_optional_bool_uses_boolean_optional(self):
        type_fn, is_bool, skip = _resolve_cli_type(bool | None, default=None)
        assert is_bool
        assert not skip

    def test_optional_list_wraps_none_tokens(self):
        type_fn, _, _ = _resolve_cli_type(list | None, default=None)
        assert type_fn is not None
        assert type_fn("None") is None
        assert type_fn("[1, 2, 3]") == [1, 2, 3]

    def test_complex_union_still_skipped(self):
        type_fn, is_bool, skip = _resolve_cli_type(int | bytes, default=0)
        assert skip or type_fn is None
