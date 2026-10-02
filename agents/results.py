"""Explicit table results, export, and evidence gates shared by all modes."""
import io
import json

import pandas as pd


def build_result(question, plan, sql, parameters, columns, rows, truncated=False):
    aggregates = {agg.get("as") or f"{agg['op']}_{str(agg.get('column', agg['op'])).replace('.', '_')}": agg
                  for agg in plan.get("aggregates", [])}
    metadata = []
    for i, name in enumerate(columns):
        values = [row[i] for row in rows if row[i] is not None]
        agg = aggregates.get(name)
        dtype = "number" if agg or values and all(isinstance(v, (int, float)) for v in values) else "text"
        role = "metric" if agg or name in {"volume", "demographic_percentage"} else "dimension"
        if not agg and (name.endswith("_id") or name in {"user_id", "tweet_id", "author_id"}):
            role = "identifier"
        if name == "prediction_month":
            role = "period"
        source = str(agg.get("column", "") if agg else name).split(".")[-1]
        unit = "adet" if agg and agg["op"] in {"count", "count_distinct"} or source == "volume" else "%" if source == "demographic_percentage" else "değer"
        metadata.append({"name": name, "type": dtype, "role": role, "unit": unit,
                         "operation": agg.get("op") if agg else None,
                         "source_column": agg.get("column") if agg else name})
    count_positions = [i for i, m in enumerate(metadata) if m["operation"] in {"count", "count_distinct"}]
    no_population = (bool(count_positions) and all(row[i] == 0 for row in rows for i in count_positions)) or not any(value is not None for row in rows for value in row)
    warnings = []
    if truncated:
        warnings.append("Sonuç satır sınırı nedeniyle kesildi; tablo tam dağılımı temsil etmeyebilir.")
    if plan.get("table") == "trending_topics" or any(j.get("table") == "trending_topics" for j in plan.get("joins", [])):
        warnings.append("Bir tweet birden fazla konu taşır. Konu hacimleri farklı tweet veya kullanıcı toplamı değildir; grup payları örtüşebilir.")
    if plan.get("joins"):
        warnings.append("JOIN sonrası satır sayısı analiz birimi sayısı değildir; farklı tweet/kullanıcı metriklerini kontrol edin.")
    return {"question": question, "json_query": plan, "sql": sql, "parameters": parameters,
            "result": str(rows), "columns": columns, "rows": [list(row) for row in rows],
            "metadata": metadata, "status": "empty" if not rows or no_population else "success",
            "truncated": truncated, "warnings": warnings, "filters": plan.get("filters", []),
            "grain": plan.get("group_by", []) or ["tek toplam" if aggregates else "kayıt"],
            "row_count": len(rows)}


def dataframe(result):
    return pd.DataFrame(result["rows"], columns=result["columns"])


def require_evidence(results):
    usable = [r for r in results if r.get("status") == "success" and r.get("rows") and not r.get("truncated")]
    if not usable:
        raise ValueError("Yeterli kanıt yok: sorgular boş, başarısız veya kesilmiş. Nihai yorum üretilmedi.")
    return usable


def evidence_text(results):
    usable = require_evidence(results)
    evidence = [{"question": r["question"], "columns": r["columns"], "rows": r["rows"],
                 "filters": r.get("filters", []), "warnings": r.get("warnings", [])} for r in usable]
    return json.dumps(evidence, ensure_ascii=False, default=str)


def export_csv(frame):
    # Spreadsheet programs may execute text beginning with these characters.
    safe = frame.copy()
    safe.columns = ["'"+str(col) if str(col).lstrip().startswith(("=", "+", "-", "@")) else col for col in safe.columns]
    for name in safe.columns:
        safe[name] = safe[name].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
    return safe.to_csv(index=False).encode("utf-8-sig")


def export_excel(frame):
    safe = frame.copy()
    safe.columns = ["'"+str(col) if str(col).lstrip().startswith(("=", "+", "-", "@")) else col for col in safe.columns]
    for name in safe.columns:
        safe[name] = safe[name].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        safe.to_excel(writer, index=False, sheet_name="Veri")
    return output.getvalue()
