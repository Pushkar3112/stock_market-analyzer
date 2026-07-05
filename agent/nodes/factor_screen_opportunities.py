"""
factor_screen_opportunities node — quantitative factor screening.
LLM explains WHY candidates passed. It does NOT pick them.
"""
import uuid
import logging
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PortfolioState
from ..config import config
from ..prompts import FACTOR_SCREENING_PROMPT
from ...mcp_server.tools.factor_screen import factor_screen as _factor_screen

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0.1,
    max_tokens=1500,
)


async def factor_screen_opportunities(state: PortfolioState) -> dict:
    """
    1. Run quantitative factor_screen tool (value + momentum + quality).
    2. Use LLM to explain why each candidate passed — not to pick stocks.
    """
    tool_call_registry = dict(state.get("tool_call_registry", {}))
    errors = []

    call_id = f"factor_screen_{uuid.uuid4().hex[:8]}"
    try:
        screen_result = await _factor_screen(
            universe=[],  # full default universe
            factors=["momentum", "quality", "low_volatility"],
            top_n=10,
        )
        tool_call_registry[call_id] = {
            "tool": "factor_screen",
            "args": {"factors": ["momentum", "quality", "low_volatility"]},
            "output": screen_result,
            "timestamp": datetime.now().isoformat(),
        }

        candidates = screen_result.get("candidates", [])
        logger.info(f"Factor screen returned {len(candidates)} candidates [call_id={call_id}]")

        # LLM explains each candidate — does NOT invent new ones
        enriched_candidates = []
        if candidates:
            for cand in candidates[:5]:  # top 5 for LLM explanation
                prompt = (
                    f"Symbol: {cand['symbol']}\n"
                    f"Sector: {cand['sector']}\n"
                    f"Composite Score: {cand['composite_score']}\n"
                    f"Factor Scores: {cand['factor_scores']}\n\n"
                    "Explain in 2-3 sentences why this stock passed the quantitative factor screen. "
                    "Do NOT make price predictions. Do NOT recommend buying or selling."
                )
                try:
                    response = await llm.ainvoke([
                        SystemMessage(content=FACTOR_SCREENING_PROMPT),
                        HumanMessage(content=prompt),
                    ])
                    explanation = response.content
                except Exception as llm_err:
                    explanation = f"LLM explanation unavailable: {llm_err}"

                enriched_candidates.append({
                    **cand,
                    "explanation": explanation,
                    "source_tool_call_id": call_id,
                })

        # Remaining candidates without LLM explanation
        for cand in candidates[5:]:
            enriched_candidates.append({**cand, "explanation": None, "source_tool_call_id": call_id})

        return {
            "screened_opportunities": enriched_candidates,
            "tool_call_registry": tool_call_registry,
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Factor screening failed: {str(e)}")
        logger.error(f"Factor screen error: {e}")
        return {
            "screened_opportunities": [],
            "tool_call_registry": tool_call_registry,
            "errors": errors,
        }
