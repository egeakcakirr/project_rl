"""Smoke test for the generated algorithm module.

Verifies only that ``algorithm.py`` is importable. The shape of the public API
is decided by the LLM and the user's task; do not assert specific symbol names
here so the suite stays stable across regenerations.
"""

import algorithm


def test_algorithm_module_imports() -> None:
    assert algorithm is not None
