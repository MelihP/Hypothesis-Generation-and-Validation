# System Prompts Architecture & Engineering Guide

This document outlines the organization of system prompts in [`agents/prompts/`] and specifies the necessary additions and review areas for ClickHouse schema compatibility and few-shot guidance.

> **Note on Current State:**
> The system prompts have been centralized into dedicated modules. Newly integrated and adapted prompts should be reviewed and verified to ensure full alignment with database schemas and model execution requirements.

---

## 1. Prompt Modules & Reference Map

The prompt architecture is organized into four functional modules:

```
agents/prompts/
├── domain_prompts.py      # Business logic, marketing dictionary, and heuristic rules
├── query_prompts.py       # SPJQ JSON query generation and SQL agent prefix
├── rewrite_prompts.py     # Contextual threading, intent decomposition, and hypothesis formation
└── synthesis_prompts.py   # CMO executive summaries, scorecards, and predictive insights
```

### A. Domain & Marketing Rules (`domain_prompts.py`)
Encapsulates business and marketing logic to ensure empirical interpretation of SQL results.

* **`MARKETING_CONCEPT_DEFINITIONS`**: Standardizes core consumer journey stages (`Consideration`, `Purchase`, `Recommendation`, `Complaint`) and causal brand health dimensions (`Solution`, `Reason to Believe`, Demographics).
* **`BUSINESS_HEURISTIC_RULES`**: Strategic inference rules instructing the LLM to:
  * Interpret zero/null consideration volume as **top-of-funnel narrowing** rather than "missing data".
  * Resolve trend divergences (e.g., falling recommendations with stable purchases indicate a necessary "Solution" brand).
  * Enforce 3-tier evidence grading: `[Doğrulandı / Demonstrated]`, `[Makul / Plausible]`, `[Desteklenmeyen / Unsupported]`.
* **`get_domain_context_prompt()`**: Injects the marketing dictionary and heuristic rules into downstream synthesis chains.
* **`build_hypothesis_synthesis_prompt()`**: Template for single-hypothesis verification against SQL evidence.

---

### B. Query Planning & SQL Prompts (`query_prompts.py`)
Guides the translation of natural language questions into structured SPJQ (Select-Project-Join-Query) JSON objects.

* **`QUERY_GENERATOR_SYSTEM_PROMPT`**: System prompt providing the SPJQ JSON grammar (`table`, `joins`, `columns`, `aggregates`, `filters`, `group_by`, `order_by`, `limit`) and noise-reduction rules for aggregating predictions and topics.
* **`SQL_AGENT_PREFIX`**: System prefix for LangChain's interactive SQL agent (`create_sql_agent`), enforcing operational root cause discovery and demographic JOINs (`is_org = 0`).

---

### C. Rewrite, Threading & Decomposition Prompts (`rewrite_prompts.py`)
Handles conversational context resolution, user clarification, and multi-hypothesis formulation.

* **`CONTEXTUALIZE_QUERY_PROMPT`**: Resolves conversational follow-up questions (e.g., *"How about among women?"*) into self-contained SQL research questions while preserving historical context and filters.
* **`ASSESS_CLARIFICATION_NEED_PROMPT`**: Evaluates query ambiguity and outputs structured interactive clarification options when user requests are too broad.
* **`MACRO_QUESTION_PROMPT`**: Generates a high-level strategic research question based on a summary of database contents.
* **`DECOMPOSE_QUESTION_PROMPT`**: Breaks down a macro strategic question into two targeted SQL sub-questions.
* **`COMPETING_HYPOTHESES_PROMPT`**: Formulates a tri-hypothesis framework in JSON:
  * $H_0$: Null Hypothesis (general market trend / external factor)
  * $H_1$: Primary Hypothesis (internal friction / top-of-funnel collapse)
  * $H_2$: Competing Hypothesis (direct-to-purchase / channel migration)
  alongside 2 SQL verification sub-questions.
* **`PREDICTIVE_TRENDS_PROMPT`**: Decomposes a forecasting request into historical time-series SQL sub-queries.

---

### D. Strategic Synthesis & Executive Insights (`synthesis_prompts.py`)
Transforms raw SQL execution results into executive-level briefings and comparative decision scorecards.

* **`EXECUTIVE_SUMMARY_PROMPT`**: CMO synthesis prompt creating a concise 3–4 sentence executive summary and 1 concrete strategic action.
* **`COMPETING_HYPOTHESES_EVALUATION_PROMPT`**: Econometrician and CMO scorecard prompt producing a comparative Markdown decision table (`[KABUL]`, `[KISMEN]`, `[REDDEDİLDİ]`), concrete metric citations, winning hypothesis analysis, and a 2-point action plan.
* **`PREDICTIVE_INSIGHT_PROMPT`**: Growth strategist prompt projecting future opportunities, risks, and proactive mitigation steps based on time-series evidence.

---

## 2. Necessary Changes & Upgrades

### 1. Critical Issues: Legacy Schema & Column Hallucination

> [!CAUTION]
> **High Priority Fix Required**: The prompt templates and fallback routines contain hardcoded references to legacy SQLite demo columns (`topic_categories`, `products`) and demo tables (`twitter_tweets`, `demo_brand_users`). These legacy references instruct the LLM to generate invalid queries that cause ClickHouse execution exceptions (`DB::Exception: Unknown expression identifier topic_categories` and `Table default.twitter_tweets doesn't exist`).

#### Exact Locations in Codebase:
1. **`agents/prompts/query_prompts.py`**:
   - **Lines 40–43 (`QUERY_GENERATOR_SYSTEM_PROMPT`)**: Instructs grouping by non-existent columns `'topic_categories'` or `'products'`.
   - **Lines 45–52 (`QUERY_GENERATOR_SYSTEM_PROMPT`)**: Instructs table `'twitter_tweets'` and joins on `'demo_brand_users'` with `'twitter_tweets.author_id = demo_brand_users.id'`.
   - **Lines 53–55 (`QUERY_GENERATOR_SYSTEM_PROMPT`)**: Instructs `LIKE` filtering on `'twitter_tweets.consumer_journey'` instead of ClickHouse task-based array filtering.
   - **Lines 66–67 (`SQL_AGENT_PREFIX`)**: References `'topics', 'topic_categories', 'products'` and `'twitter_tweets'` JOIN `'demo_brand_users'`.
2. **`agents/prompts/rewrite_prompts.py`**:
   - **Lines 68–69 (`DECOMPOSE_QUESTION_PROMPT`)**: Instructs sub-questions to group by `'topic_categories'` or `'products'` and join `'twitter_tweets'`.
   - **Line 89 (`COMPETING_HYPOTHESES_PROMPT`)**: Few-shot example test question includes `"topic_categories"`.
3. **`agents/rewrite_nl_agent.py`**:
   - **Line 121 (`formulate_competing_hypotheses`)**: Fallback question list uses `"topic_categories"`.
4. **`app.py`**:
   - **Line 251 (`MOD 1: Otonom İçgörü`)**: Passes obsolete schema string `"Tablolar: twitter_tweets, demo_brand_users, demo_brand_predictions"` to `generate_macro_question`.
5. **`tests/`**:
   - `tests/test_rewrite_nl_agent.py` (Lines 44, 65): Assertions expecting `"topic_categories"`.
   - `tests/test_query_compiler.py` (Lines 105–117): Joins referencing `twitter_tweets` and `demo_brand_users`.

#### ClickHouse Schema Alignment Mapping:
| Legacy Prompt Pattern | Target ClickHouse Schema Pattern |
| :--- | :--- |
| `group_by: ["topic_categories"]` | Filter: `task_name = 'topic_monthly'`, column: `category_value` |
| `group_by: ["products"]` | Filter: `task_name = 'brand'` / `brand_sector`, column: `category_value` |
| `twitter_tweets.consumer_journey LIKE '...'` | Filter: `task_name = 'consumer_journey'` AND `has(category_value, '...')` |
| `table: 'twitter_tweets'` | `table: 'tweet_predictions'` or `table: 'tweets'` |
| `joins: [{'table': 'demo_brand_users'}]` | `joins: [{'table': 'users', 'on': {'left': 'tweets.author_id', 'right': 'users.id'}}]` |

---

### 2. ClickHouse Schema Compatibility & Semantic Mapping
To ensure accurate query planning against ClickHouse, the system prompts require explicit schema context and querying semantics:

* **Table Purpose Definitions**:
  * `tweets`: Raw post content, creation timestamps, engagement counts, and author identifiers.
  * `tweet_predictions`: Model-predicted classifications including consumer journey stages, sentiment, topics, topic categories, and product mentions.
  * `users`: Account demographic data (e.g., `age_range`, `gender`, `location`, `is_org`).
  * `user_factors`: Aggregated user-level psychographic and behavioral factor scores.
* **Relational Joins & Foreign Keys**:
  * Joining `tweets` with `tweet_predictions` on `tweet_id`.
  * Joining `tweets` with `users` on `tweets.author_id = users.id`.
  * Filtering non-human / organization accounts via `users.is_org = 0`.
* **ClickHouse-Specific Operations**:
  * Querying `Array(String)` columns using `HAS`, `HAS_ANY`, and `HAS_ALL`.
  * Flattening nested arrays using `ARRAY JOIN`.
  * Aggregating distinct array values via `groupArray`.

---

### 3. Few-Shot Example Integration
Adding concrete input-output examples directly inside the prompts ensures consistent formatting and prevents syntax hallucinations:

* **`QUERY_GENERATOR_SYSTEM_PROMPT`**:
  * Examples covering grouped aggregations with ordering and limits (e.g., top complaint categories in a specific timeframe using `task_name = 'consumer_journey'`).
  * Examples demonstrating multi-table demographic JOINs between `tweets`, `tweet_predictions`, and `users` with `is_org = 0` bot exclusion.
  * Examples showing array containment filtering on `consumer_journey` or `topics` with `has()`.
* **`COMPETING_HYPOTHESES_PROMPT`**:
  * Reference examples mapping ambiguous business observations into distinct $H_0, H_1, H_2$ formulations and valid ClickHouse-compatible verification questions.
* **`ASSESS_CLARIFICATION_NEED_PROMPT`**:
  * Examples distinguishing between specific actionable queries (no clarification needed) and broad/vague queries (generating category/product option cards).
