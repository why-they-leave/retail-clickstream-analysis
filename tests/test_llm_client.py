"""llm_connector.client tests."""

import pytest

from src.llm_connector.client import run_parallel


def _maybe_fail(value):
    if value == "bad":
        raise ValueError("boom")
    return value.upper()


def _fallback(error, task_args):
    return {"arg": task_args[0], "error": str(error)}


def test_run_parallel_raises_without_error_handler():
    with pytest.raises(ValueError, match="boom"):
        run_parallel(_maybe_fail, [("ok",), ("bad",)], max_workers=1)


def test_run_parallel_uses_error_handler_and_continues():
    results = run_parallel(
        _maybe_fail,
        [("ok",), ("bad",), ("done",)],
        max_workers=1,
        on_error=_fallback,
    )

    assert "OK" in results
    assert "DONE" in results
    assert {"arg": "bad", "error": "boom"} in results
    assert len(results) == 3
