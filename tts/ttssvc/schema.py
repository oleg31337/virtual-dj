"""Validation and normalisation of engine-declared synthesis parameters.

The service never hardcodes a parameter key: an engine declares a schema
(``params()``) and this module turns a caller's ``{key: value}`` into values the
engine can trust, reporting what it had to ignore or clamp.

Deliberately LENIENT, because the caller keeps a saved ``params`` dict that may
predate the current engine: an unknown key is ignored (reported), an
out-of-range value is clamped (reported), and only a wrong *type* is an error —
that is a real client bug, not a stale setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NUMERIC_TYPES = ("float", "int")


class ParamError(ValueError):
    """A value whose type cannot be coerced into the declared parameter."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(f"{key}: {message}")
        self.key = key
        self.message = message


@dataclass
class Resolved:
    values: dict[str, Any] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)


def defaults(schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Every declared parameter at its default value."""
    return {p["key"]: p.get("default") for p in schema if p.get("key")}


def _coerce_number(key: str, spec: dict[str, Any], value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ParamError(key, f"expected a number, got {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParamError(key, f"expected a number, got {value!r}") from exc
    if spec.get("type") == "int":
        return int(round(number))
    return float(number)


def _clamp(spec: dict[str, Any], value: float | int) -> tuple[float | int, bool]:
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and value < lo:
        value = lo
        return value, True
    if hi is not None and value > hi:
        value = hi
        return value, True
    return value, False


def resolve(schema: list[dict[str, Any]], given: dict[str, Any] | None,
            *, strict_types: bool = True) -> Resolved:
    """Merge ``given`` over the schema defaults, validating and clamping."""
    out = Resolved(values=defaults(schema))
    given = given or {}
    by_key = {p["key"]: p for p in schema if p.get("key")}

    for key, raw in given.items():
        spec = by_key.get(key)
        if spec is None:
            out.ignored.append(key)
            continue
        if raw is None:
            continue
        kind = spec.get("type", "float")

        if kind == "bool":
            if isinstance(raw, bool):
                out.values[key] = raw
            elif isinstance(raw, (int, str)) and str(raw).lower() in (
                    "true", "false", "1", "0", "yes", "no"):
                out.values[key] = str(raw).lower() in ("true", "1", "yes")
            elif strict_types:
                raise ParamError(key, f"expected a boolean, got {raw!r}")
            continue

        if kind == "enum":
            choices = [c.get("value") if isinstance(c, dict) else c
                       for c in (spec.get("choices") or [])]
            if raw in choices:
                out.values[key] = raw
            else:
                # A stale choice from another engine is a stale setting, not a
                # client bug: keep the declared default and say it was dropped.
                out.ignored.append(key)
            continue

        if kind in NUMERIC_TYPES:
            number = _coerce_number(key, spec, raw)
            number, was_clamped = _clamp(spec, number)
            out.values[key] = number
            if was_clamped:
                out.clamped.append(key)
            continue

        # Unknown declared type: pass through untouched (forward compatibility).
        out.values[key] = raw

    return out
