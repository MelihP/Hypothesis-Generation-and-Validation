"""Non-LLM table building, full-table quality checks, and independent-user tests."""
from agents.analysis_service import dataset_catalog
from agents.query_agent import quote_ident
from agents.sql_safety import database_path, read_query
from agents.statistics import categorical_test, wilson_interval
import pandas as pd


def quality_report(query, table):
    catalog = dataset_catalog(query)
    if table not in catalog:
        raise ValueError("Tablo bulunamadı.")
    columns = catalog[table]["columns"]
    expressions = ["count(*)"] + [f"count({quote_ident(col)})" for col in columns]
    _, rows, _ = read_query(database_path(query.db), f"SELECT {', '.join(expressions)} FROM {quote_ident(table)}", allowed_tables={table})
    total, *valid = rows[0]
    return pd.DataFrame([{"Sütun": col, "Toplam": total, "NULL": total-count,
                           "NULL (%)": (total-count)/total*100 if total else None}
                          for col, count in zip(columns, valid)])


def table_plan(table, groups, metric="records", filters=None, limit=200):
    metrics = {"records": {"op": "count", "as": "kayıt_sayısı"},
               "tweets": {"op": "count_distinct", "column": "tweet_id", "as": "farklı_tweet"},
               "users": {"op": "count_distinct", "column": "user_id" if table=="demographics" else "author_id", "as": "farklı_kullanıcı"},
               "volume": {"op": "sum", "column": "volume", "as": "toplam_hacim"}}
    if metric not in metrics:
        raise ValueError("Metrik desteklenmiyor.")
    agg = metrics[metric]
    return {"table": table, "group_by": groups, "aggregates": [agg], "filters": filters or [],
            "order_by": [{"column": agg["as"], "dir": "desc"}], "limit": limit}


def independent_user_comparison(query, table, dimension, period=None):
    outcomes = {"consumer_journey": "journey_stage", "emotion_analysis": "dominant_emotion"}
    if table not in outcomes or dimension not in {"age_group", "gender"}:
        raise ValueError("Bu karşılaştırma yalnızca yaş/cinsiyet ve yolculuk/duygu için desteklenir.")
    outcome = outcomes[table]
    params = [period] if period is not None else []
    where = "WHERE prediction_month = ?" if period is not None else ""
    # Users with multiple observations are excluded, so no user contributes to multiple cells.
    sql = f"""WITH eligible AS (
        SELECT author_id, min({quote_ident(outcome)}) AS outcome FROM {quote_ident(table)} {where}
        GROUP BY author_id HAVING count(*) = 1)
        SELECT d.{quote_ident(dimension)} AS demographic, e.outcome, count(*) AS users
        FROM eligible e JOIN demographics d ON d.user_id = e.author_id
        WHERE d.{quote_ident(dimension)} IS NOT NULL AND e.outcome IS NOT NULL
        AND trim(d.{quote_ident(dimension)}) != '' AND trim(e.outcome) != ''
        GROUP BY d.{quote_ident(dimension)}, e.outcome ORDER BY demographic, outcome"""
    path = database_path(query.db)
    _, duplicate, _ = read_query(path, "SELECT count(*) FROM (SELECT user_id FROM demographics GROUP BY user_id HAVING count(*) > 1)", allowed_tables={"demographics"})
    if duplicate[0][0]:
        raise ValueError("Demografik anahtarlar tekil değil; bağımsız örneklem oluşturulamadı.")
    _, rows, truncated = read_query(path, sql, params, allowed_tables={table, "demographics", "eligible"})
    if truncated or not rows:
        raise ValueError("Karşılaştırma tablosu boş veya kesilmiş.")
    frame = pd.DataFrame(rows, columns=[dimension, outcome, "users"])
    cross = frame.pivot(index=dimension, columns=outcome, values="users").fillna(0).astype(int)
    test = categorical_test(cross.values, independent=True)
    proportions = []
    for group, row in cross.iterrows():
        total = int(row.sum())
        for label, count in row.items():
            low, high = wilson_interval(int(count), total)
            proportions.append({"group": str(group), "outcome": str(label), "users": int(count),
                                "denominator": total, "proportion": int(count)/total,
                                "ci95_wilson": [low, high]})
    test["group_proportions"] = proportions
    test["interval_scope"] = "Grup içi kullanıcı oranı için bireysel %95 Wilson aralıkları; eşzamanlı/multiplicity düzeltmesi yok."
    _, totals, _ = read_query(path, f"SELECT count(DISTINCT author_id) FROM {quote_ident(table)} {where}", params, allowed_tables={table})
    test["scope"] = {"period": period, "total_authors": totals[0][0], "included_users": int(cross.values.sum()),
                      "excluded_users": totals[0][0]-int(cross.values.sum()),
                      "note": "Birden fazla kaydı olan, eşleşmeyen veya etiketi eksik kullanıcılar hariç tutuldu. Seçim yanlılığı ve demografik/duygu tahmin hataları olabilir. Tek keşifsel testtir; çoklu test düzeltmesi uygulanmadı."}
    return cross, test
