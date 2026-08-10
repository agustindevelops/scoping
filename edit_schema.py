"""Load and validate workspace edit configs against the in-repo JSON schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "edit_config.schema.json"

_validator: Draft202012Validator | None = None


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def get_validator() -> Draft202012Validator:
    global _validator
    if _validator is None:
        _validator = Draft202012Validator(load_schema())
    return _validator


def format_validation_error(exc: ValidationError) -> str:
    path = ".".join(str(part) for part in exc.absolute_path) or "(root)"
    return f"{path}: {exc.message}"


def validate_edit_config(data: Any, *, source: str | Path | None = None) -> list[dict[str, Any]]:
    """Validate and return the list of edit jobs. Raises ValueError on failure."""
    validator = get_validator()
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        details = "; ".join(format_validation_error(err) for err in errors[:8])
        prefix = f"{source}: " if source else ""
        more = f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""
        raise ValueError(f"{prefix}invalid edit config — {details}{more}")

    if not isinstance(data, list) or not data:
        raise ValueError(f"{source or 'config'}: root must be a non-empty array of jobs")
    return data


def load_edit_config(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return validate_edit_config(raw, source=path)
