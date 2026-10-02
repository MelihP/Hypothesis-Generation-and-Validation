"""
Sistem Promptları ve Alan Kuralları Modülü (Agents Prompts Package).
Tüm LLM sistem promptlarını, şablonlarını ve alan kurallarını tek bir noktada toplar.
"""

from agents.prompts.domain_prompts import (
    MARKETING_CONCEPT_DEFINITIONS,
    BUSINESS_HEURISTIC_RULES,
    get_domain_context_prompt,
    build_hypothesis_synthesis_prompt,
)

from agents.prompts.query_prompts import (
    QUERY_GENERATOR_SYSTEM_PROMPT,
    SQL_AGENT_PREFIX,
)

from agents.prompts.rewrite_prompts import (
    CONTEXTUALIZE_QUERY_PROMPT,
    ASSESS_CLARIFICATION_NEED_PROMPT,
    MACRO_QUESTION_PROMPT,
    DECOMPOSE_QUESTION_PROMPT,
    COMPETING_HYPOTHESES_PROMPT,
    PREDICTIVE_TRENDS_PROMPT,
)

from agents.prompts.synthesis_prompts import (
    EXECUTIVE_SUMMARY_PROMPT,
    COMPETING_HYPOTHESES_EVALUATION_PROMPT,
    PREDICTIVE_INSIGHT_PROMPT,
)

__all__ = [
    # Domain & Marketing Rules
    "MARKETING_CONCEPT_DEFINITIONS",
    "BUSINESS_HEURISTIC_RULES",
    "get_domain_context_prompt",
    "build_hypothesis_synthesis_prompt",
    # Query & SQL Agent
    "QUERY_GENERATOR_SYSTEM_PROMPT",
    "SQL_AGENT_PREFIX",
    # Rewrite & Decomposition
    "CONTEXTUALIZE_QUERY_PROMPT",
    "ASSESS_CLARIFICATION_NEED_PROMPT",
    "MACRO_QUESTION_PROMPT",
    "DECOMPOSE_QUESTION_PROMPT",
    "COMPETING_HYPOTHESES_PROMPT",
    "PREDICTIVE_TRENDS_PROMPT",
    # Synthesis & Insights
    "EXECUTIVE_SUMMARY_PROMPT",
    "COMPETING_HYPOTHESES_EVALUATION_PROMPT",
    "PREDICTIVE_INSIGHT_PROMPT",
]
