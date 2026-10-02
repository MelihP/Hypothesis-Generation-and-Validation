"""Explicit opt-in live test (uses OpenAI credits); never logs credentials."""
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import get_hybrid_agent
from agents.analysis_service import AnalysisService


def main():
    logging.disable(logging.CRITICAL)
    path = Path(__file__).resolve().parents[1]/"insight_generation_bot.db"
    _, _, docs, query, rewrite, synthesis = get_hybrid_agent(db_uri=f"sqlite:///{path}")
    table = query.execute_nl_query("demographics tablosunda yaş gruplarına göre farklı user_id sayısını getir. Tüm grupları kapsa, filtre uygulama.")
    assert table["status"] == "success" and not table["truncated"]
    expected = query.execute_plan({"table": "demographics", "aggregates": [{"op": "count_distinct", "column": "user_id", "as": "users"}]})
    assert sum(row[1] for row in table["rows"]) == expected["rows"][0][0]
    print("LIVE_TABLE: PASS", flush=True)
    answer = AnalysisService(query, rewrite, synthesis, docs).analyze("demographics tablosunda toplam kullanıcı sayısı nedir?", source="auto")
    assert answer["status"] == "success" and answer["insight"], answer.get("error", "Incomplete evidence")
    print("LIVE_ROUTED_ANALYSIS: PASS", flush=True)
    resolved = rewrite.contextualize_query("Peki kadınlarda nasıl?", [{"user": "Yolculuk tweet sayısı", "context": [
        {"table": "consumer_journey", "filters": [{"column": "prediction_month", "op": "EQ", "value": 202607}]}]}])
    assert "202607" in resolved or "2026" in resolved
    print("LIVE_MEMORY: PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"LIVE_FAILURE: {type(exc).__name__}; status={getattr(exc, 'status_code', None)}", flush=True)
        sys.exit(1)
