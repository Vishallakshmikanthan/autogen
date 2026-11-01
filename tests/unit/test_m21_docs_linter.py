"""Unit tests for Documentation and Algorithm Registry Linter (SYS-19, Module M21)."""

from scripts.docs_lint import lint_documentation


def test_docs_lint_passes_on_repository():
    """Verify that all registered algorithm documents in the repo pass documentation linting."""
    error_count, errors = lint_documentation()
    assert error_count == 0, f"Docs lint produced errors: {errors}"
