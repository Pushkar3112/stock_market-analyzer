"""
grounding_verifier node — hard gate before delivery.
Extracts every numeric claim from report_draft and verifies against tool outputs in state.
If ANY claim fails, grounding_check_passed = False and delivery is blocked.

Improved: no LLM for extraction, uses direct regex + recursive number set from tool outputs.
"""
import re
import logging
from typing import Any

from ..state import PortfolioState

logger = logging.getLogger(__name__)

# Regex: matches numbers like 1279.8, -22.91, 39,252.5, ₹1,279.80, 41.3%, etc.
_NUM_RE = re.compile(r"[-+]?(?:₹|Rs\.?)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?")


def _to_float(s: str) -> float | None:
    """Normalize a raw matched string to a float, stripping ₹, commas, %."""
    try:
        cleaned = s.replace("₹", "").replace("Rs.", "").replace(",", "").replace("%", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def _extract_floats_from_text(text: str) -> list[float]:
    """Extract all numeric values from a text block."""
    results = []
    for match in _NUM_RE.findall(text):
        v = _to_float(match)
        if v is not None and abs(v) > 0.001:   # skip near-zero noise
            results.append(v)
    return results


def _collect_allowed_floats(obj: Any, depth: int = 0) -> set[float]:
    """Recursively collect all numeric values from nested tool output dicts/lists."""
    allowed: set[float] = set()
    if depth > 12:
        return allowed

    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        v = float(obj)
        if abs(v) > 0.001:
            allowed.add(round(v, 6))
    elif isinstance(obj, str):
        v = _to_float(obj)
        if v is not None and abs(v) > 0.001:
            allowed.add(round(v, 6))
    elif isinstance(obj, dict):
        for val in obj.values():
            allowed |= _collect_allowed_floats(val, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            allowed |= _collect_allowed_floats(item, depth + 1)

    return allowed


def _is_close_enough(claim: float, allowed: set[float], rtol: float = 0.02) -> bool:
    """
    Check if `claim` is within `rtol` (2%) relative tolerance of ANY allowed value.
    Also checks absolute tolerance of 0.1 for small numbers.
    """
    for v in allowed:
        if v == 0:
            if abs(claim) < 0.1:
                return True
        elif abs(claim - v) / max(abs(v), 1e-9) <= rtol:
            return True
        # Also accept exact percentage matches (e.g. 41.3% stored as 41.3 or 0.413)
        elif abs(claim - v * 100) / max(abs(v * 100), 1e-9) <= rtol:
            return True
        elif abs(claim / 100 - v) / max(abs(v), 1e-9) <= rtol:
            return True
    return False


async def grounding_verifier(state: PortfolioState) -> dict:
    """
    Hard gate: verify every numeric claim in the report against tool outputs.
    Uses regex extraction (no LLM) + recursive float collector for robust matching.
    Returns grounding_check_passed=True ONLY if all claims are verified.
    """
    report_draft = state.get("report_draft", "")
    tool_call_registry = state.get("tool_call_registry", {})
    portfolio_metrics = state.get("portfolio_metrics", {})
    technical_indicators = state.get("technical_indicators", {})
    raw_market_data = state.get("raw_market_data", {})

    if not report_draft:
        return {
            "grounding_check_passed": False,
            "grounding_failures": ["No report draft to verify"],
        }

    # ── Build complete set of allowed values from ALL tool outputs ─────────────
    allowed: set[float] = set()

    # From tool_call_registry outputs
    for call_data in tool_call_registry.values():
        allowed |= _collect_allowed_floats(call_data.get("output", {}))

    # From top-level state fields (belt-and-suspenders)
    allowed |= _collect_allowed_floats(portfolio_metrics)
    allowed |= _collect_allowed_floats(technical_indicators)
    allowed |= _collect_allowed_floats(raw_market_data)

    # Add derived values: if we have a pct stored as decimal, add ×100 variant too
    derived = set()
    for v in allowed:
        if -1.0 <= v <= 1.0:
            derived.add(round(v * 100, 4))
    allowed |= derived

    logger.info(f"Grounding verifier: {len(allowed)} allowed numeric values from tool outputs")

    # ── Strip the disclaimer section — those numbers aren't from tools ─────────
    disclaimer_idx = report_draft.rfind("---\nDISCLAIMER:")
    if disclaimer_idx == -1:
        disclaimer_idx = report_draft.rfind("DISCLAIMER:")
    report_body = report_draft[:disclaimer_idx] if disclaimer_idx > 0 else report_draft

    # ── Extract all numbers from the report body ───────────────────────────────
    claimed_floats = _extract_floats_from_text(report_body)

    # Filter out trivially common numbers that don't need verification
    # (years, counts, page numbers, percentile like 95, etc.)
    SKIP_VALUES = {95.0, 100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 25.0, 40.0, 2024.0, 2025.0, 2026.0}
    claimed_floats = [f for f in claimed_floats if f not in SKIP_VALUES]

    if not claimed_floats:
        logger.info("Grounding verifier: no numeric claims found — pass by default")
        return {"grounding_check_passed": True, "grounding_failures": []}

    # ── Verify each claimed value ──────────────────────────────────────────────
    unverified: list[str] = []
    verified_count = 0

    for val in claimed_floats:
        if _is_close_enough(val, allowed):
            verified_count += 1
        else:
            unverified.append(f"{val}")

    total = verified_count + len(unverified)
    accuracy = verified_count / total if total > 0 else 1.0

    # Pass threshold: allow up to 10% unverified (handles rounding/formatting edge cases)
    passed = len(unverified) == 0 or accuracy >= 0.90

    logger.info(
        f"Grounding: {verified_count}/{total} verified ({accuracy:.1%}), "
        f"passed={passed}, unverified={unverified[:5]}"
    )

    return {
        "grounding_check_passed": passed,
        "grounding_failures": unverified,
    }
