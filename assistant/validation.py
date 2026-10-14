"""Validation guardrail.

Before an answer is shown, check that:
  1. every number it quotes (percentages, dollar values, decimals, point changes)
     appears in the data the tools returned, and
  2. every link it contains is a report link the tools generated.

Anything that fails is flagged to the user rather than silently shown as fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

TOLERANCE = 0.05  # values are pre-rounded to 1 decimal place by the semantic layer

_URL = re.compile(r"https?://[^\s)\]\"'>`【】]+")
_MONEY = re.compile(r"\$\s?(-?[\d,]+(?:\.\d+)?)\s*([kKmM])?")
_PERCENT_OR_POINTS = re.compile(
    r"([+\-−]?\d[\d,]*(?:\.\d+)?)(?:\s?%|[\s\-‑–]?(?:pts|points?|percentage[\s\-‑]points?)\b)")
_GROUPED_INTEGER = re.compile(r"(?<![\w.$])(\d{1,3}(?:,\d{3})+)(?![\d%])")   # e.g. 102,346
_DECIMAL = re.compile(r"(?<![\w.$])([+\-−]?\d+\.\d+)(?![\w%.])")


@dataclass
class ValidationResult:
    numbers_checked: int = 0
    unmatched_numbers: list[str] = field(default_factory=list)
    unknown_links: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unmatched_numbers and not self.unknown_links

    def summary(self) -> str:
        if self.ok:
            return f"All {self.numbers_checked} numbers match the report data, and every link opens the matching report view."
        parts = []
        if self.unmatched_numbers:
            parts.append(f"{len(self.unmatched_numbers)} number(s) could not be matched to the report data: "
                         + ", ".join(self.unmatched_numbers))
        if self.unknown_links:
            parts.append(f"{len(self.unknown_links)} link(s) were not generated from the report")
        return "Check before relying on this. " + "; ".join(parts) + "."


def evidence_numbers(evidence: Iterable[Any]) -> list[float]:
    """Every numeric value found anywhere in the tool results."""
    found: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            found.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(list(evidence))
    return found


def evidence_links(evidence: Iterable[Any]) -> set[str]:
    links: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "report_link" and isinstance(v, str):
                    links.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(list(evidence))
    return links


def _to_float(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-"))


def quoted_numbers(text: str) -> list[tuple[str, float]]:
    """Numbers in the text that are claims about the data (not dates, IDs or counts of insights)."""
    text = _URL.sub(" ", text)
    out: list[tuple[str, float]] = []
    taken: list[tuple[int, int]] = []

    def add(match: re.Match, value: float):
        if any(s <= match.start() < e for s, e in taken):
            return
        taken.append(match.span())
        out.append((match.group(0).strip(), value))

    for m in _MONEY.finditer(text):
        value = _to_float(m.group(1)) * {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1)
        add(m, value)
    for m in _PERCENT_OR_POINTS.finditer(text):
        add(m, _to_float(m.group(1)))
    for m in _GROUPED_INTEGER.finditer(text):
        add(m, _to_float(m.group(1)))
    for m in _DECIMAL.finditer(text):
        add(m, _to_float(m.group(1)))
    return out


def _matches(value: float, pool: list[float], approximate: bool) -> bool:
    for ev in pool:
        for v in (value, -value):  # "fell 1.1 pts" quoting a change of -1.1
            if approximate and ev and abs(v - ev) / abs(ev) < 0.01:   # "$2.2m" for 2,245,837
                return True
            if abs(v - ev) <= TOLERANCE:
                return True
    return False


def validate(text: str, evidence: Iterable[Any]) -> ValidationResult:
    evidence = list(evidence)
    pool = evidence_numbers(evidence)
    allowed_links = evidence_links(evidence)
    result = ValidationResult()

    for token, value in quoted_numbers(text):
        result.numbers_checked += 1
        approximate = bool(re.search(r"[kKmM]\s*$", token))
        if not _matches(value, pool, approximate):
            result.unmatched_numbers.append(token)

    for url in _URL.findall(text):
        if url.rstrip(".,;") not in allowed_links:
            result.unknown_links.append(url)
    return result
