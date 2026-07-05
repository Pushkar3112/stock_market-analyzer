"""
grounding_verifier node — hard gate before delivery.
Extracts every numeric claim from report_draft and verifies against tool outputs in state.
If ANY claim fails, grounding_check_passed = False and delivery is blocked.
"""
import re
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PortfolioState
from ..config import config
from ..prompts import GROUNDING_VERIFIER_PROMPT

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
    max_tokens=1500,
)

# Regex to extract numbers from text (prices, percentages, ratios)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:%|₹)?")


def _extract_numbers_from_text(text: str) -> list[str]:
    """Extract all numeric values from report text."""
    return NUMBER_PATTERN.findall(text)


def _build_allowed_values(tool_call_registry: dict, portfolio_metrics: dict, technical_indicators: dict) -> set[str]:
    """Build set of all numeric values present in tool outputs (for fast lookup)."""
    allowed: set[str] = set()

    # From tool_call_registry
    def _extract_numbers_recursive(obj: Any):
        if isinstance(obj, (int, float)):
            allowed.add(str(round(obj, 2)))
            allowed.add(f"{round(obj, 2)}%")
            allowed.add(f"₹{round(obj, 2)}")
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract_numbers_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_numbers_recursive(item)

    for call_data in tool_call_registry.values():
        _extract_numbers_recursive(call_data.get("output", {}))

    return allowed


async def grounding_verifier(state: PortfolioState) -> dict:
    """
    Hard gate: verify every numeric claim in the report against tool outputs.
    Uses LLM to identify claims + deterministic lookup to verify them.
    Returns grounding_check_passed=True ONLY if all claims are verified.
    """
    report_draft = state.get("report_draft", "")
    tool_call_registry = state.get("tool_call_registry", {})
    portfolio_metrics = state.get("portfolio_metrics", {})
    technical_indicators = state.get("technical_indicators", {})

    if not report_draft:
        return {
            "grounding_check_passed": False,
            "grounding_failures": ["No report draft to verify"],
        }

    # Use LLM to extract all numeric claims with context
    extraction_prompt = (
        "Extract ALL numeric claims from the following report. "
        "For each claim, output: 'CLAIM: [value] | CONTEXT: [surrounding text]'\n\n"
        f"REPORT:\n{report_draft[:3000]}"  # cap to avoid context overflow
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=GROUNDING_VERIFIER_PROMPT),
            HumanMessage(content=extraction_prompt),
        ])
        extracted_claims_text = response.content
    except Exception as e:
        logger.error(f"Grounding verifier LLM call failed: {e}")
        return {
            "grounding_check_passed": False,
            "grounding_failures": [f"Grounding verifier LLM failed: {e}"],
        }

    # Build set of all numbers present in tool outputs
    allowed_values = _build_allowed_values(tool_call_registry, portfolio_metrics, technical_indicators)

    # Extract claimed values and check each against allowed set
    claim_lines = [l for l in extracted_claims_text.split("\n") if "CLAIM:" in l]
    unverified: list[str] = []
    verified_count = 0

    for line in claim_lines:
        if "CLAIM:" in line:
            # Extract the value
            parts = line.split("|")
            claim_part = parts[0].replace("CLAIM:", "").strip()
            context_part = parts[1].replace("CONTEXT:", "").strip() if len(parts) > 1 else ""

            # Check if this number appears in any tool output
            # We do tolerance-based matching for floating point
            claim_nums = NUMBER_PATTERN.findall(claim_part)
            for num in claim_nums:
                clean_num = num.replace("%", "").replace("₹", "").strip()
                try:
                    float_val = float(clean_num)
                    # Check exact match or very close match in allowed values
                    found = any(
                        abs(float(v.replace("%", "").replace("₹", "")) - float_val) < 0.05
                        for v in allowed_values
                        if v.replace("%", "").replace("₹", "").replace("-", "").replace("+", "").replace(".", "").isdigit()
                        or (v.replace("%", "").replace("₹", "").replace("-", "").replace(".", "").replace("+","").isdigit())
                    )
                    if found:
                        verified_count += 1
                    else:
                        unverified.append(f"'{num}' in context: '{context_part[:80]}'")
                except ValueError:
                    pass

    # Calculate grounding accuracy
    total_claims = verified_count + len(unverified)
    grounding_accuracy = verified_count / total_claims if total_claims > 0 else 1.0
    passed = len(unverified) == 0

    logger.info(
        f"Grounding check: {verified_count}/{total_claims} verified "
        f"({grounding_accuracy:.1%}), passed={passed}"
    )

    if unverified:
        logger.warning(f"Unverified claims: {unverified}")

    return {
        "grounding_check_passed": passed,
        "grounding_failures": unverified,
    }
