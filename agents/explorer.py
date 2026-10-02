"""Typed explorer and independent-observation statistics across both backends."""
from datetime import date
import pandas as pd
from agents.sql_compiler import quote_ident, compile_json_to_sql
from agents.data_schema import validate_query_schema
from agents.statistics import categorical_test, wilson_interval


def quality_report(query, table):
    catalog = query.catalog()
    if table not in catalog:
        raise ValueError("Tablo bulunamadı.")
    columns = catalog[table]["columns"]
    quote = lambda col: quote_ident(col, dialect=query.dialect)
    expressions = ["count(*)"] + [f"count({quote(col)})" for col in columns]
    _, rows, _ = query.read(f"SELECT {', '.join(expressions)} FROM {quote(table)}")
    total, *valid = rows[0]
    return pd.DataFrame([{"Sütun": col, "Toplam": total, "NULL": total-count,
                         "NULL (%)": (total-count)/total*100 if total else None}
                        for col, count in zip(columns, valid)])


def period_filters(info, period):
    if period is None or period == "Tümü":
        return []
    column = info.get("period_column")
    if not column:
        raise ValueError("Bu tabloda dönem alanı bulunamadı.")
    if column == "prediction_month":
        value = str(period) if "String" in info["types"][column] else int(period)
        return [{"column":column, "op":"EQ", "value":value}]
    year, month = divmod(int(period), 100)
    first = date(year, month, 1)
    last = date(year+1, 1, 1) if month==12 else date(year, month+1, 1)
    return [{"column":column, "op":"GTE", "value":str(first)},
            {"column":column, "op":"LT", "value":str(last)}]


def table_plan(table, groups, metric="records", filters=None, limit=200, info=None, expand_arrays=False):
    columns = info["columns"] if info else []
    user = next((name for name in ("user_id", "author_id", "id") if name in columns), "user_id" if table=="demographics" else "author_id")
    metrics = {"records": {"op":"count", "as":"kayıt_sayısı"},
               "tweets": {"op":"count_distinct", "column":"tweet_id", "as":"farklı_tweet"},
               "users": {"op":"count_distinct", "column":user, "as":"farklı_kullanıcı"},
               "volume": {"op":"sum", "column":"volume", "as":"toplam_hacim"}}
    if metric not in metrics:
        raise ValueError("Metrik desteklenmiyor.")
    arrays = []
    active_groups = []
    for name in groups:
        if expand_arrays and info and info["types"][name].startswith("Array("):
            alias = name + "_item"
            arrays.append({"column":name,"as":alias})
            active_groups.append(alias)
        else:
            active_groups.append(name)
    agg = metrics[metric]
    plan = {"table":table, "group_by":active_groups, "aggregates":[agg], "filters":filters or [],
            "order_by":[{"column":agg["as"],"dir":"desc"}], "limit":limit}
    if arrays:
        plan["array_joins"] = arrays
    return plan


def independent_comparison(query, table, dimension_table, unit, join_key, dimension, outcome, filters=None):
    """Conservatively exclude every unit with multiple rows or multiple labels."""
    catalog = query.catalog()
    if table not in catalog or dimension_table not in catalog:
        raise ValueError("Karşılaştırma tabloları bulunamadı.")
    plan = {"table":table, "alias":"s", "columns":[f"s.{unit}",f"d.{dimension}" if dimension_table!=table else f"s.{dimension}", f"s.{outcome}"],
            "filters":[], "joins":[]}
    for filt in filters or []:
        plan["filters"].append({**filt,"column":"s."+filt["column"]})
    if dimension_table!=table:
        plan["joins"] = [{"table":dimension_table,"alias":"d","on":{"left":f"s.{unit}","right":f"d.{join_key}"}}]
    if catalog[table]["types"].get(outcome, "").startswith("Array("):
        plan["array_joins"] = [{"column":f"s.{outcome}","as":"outcome_item"}]
        plan["columns"][2] = "outcome_item"
    if catalog[dimension_table]["types"].get(dimension, "").startswith("Array("):
        raise ValueError("İstatistiksel karşılaştırma için tek değerli demografik sütun gerekli.")
    validate_query_schema(plan, catalog)
    params = []
    sql = compile_json_to_sql(plan, dialect=query.dialect, parameters=params)
    quote = lambda name: quote_ident(name,dialect=query.dialect)
    prefix = "SELECT " + ", ".join(quote(name) for name in plan["columns"])
    if not sql.startswith(prefix):
        raise ValueError("Karşılaştırma planı oluşturulamadı.")
    select = "SELECT " + ", ".join(f"{quote(name)} AS {quote(alias)}" for name,alias in zip(plan["columns"],("observation","demographic","outcome")))
    selected = select + sql[len(prefix):]
    cross_sql = f"""WITH selected AS ({selected}), eligible AS (
        SELECT observation, min(demographic) AS demographic, min(outcome) AS outcome
        FROM selected GROUP BY observation HAVING count(*)=1)
        SELECT demographic, outcome, count(*) AS units FROM eligible
        WHERE observation IS NOT NULL AND demographic IS NOT NULL AND outcome IS NOT NULL
        GROUP BY demographic, outcome ORDER BY demographic, outcome LIMIT 1001"""
    _, rows, truncated = query.read(cross_sql, params)
    if truncated or not rows:
        raise ValueError("Karşılaştırma tablosu boş veya kesilmiş.")
    frame = pd.DataFrame(rows, columns=["demographic","outcome","units"])
    cross = frame.pivot(index="demographic",columns="outcome",values="units").fillna(0).astype(int)
    test = categorical_test(cross.values, independent=True)
    total_plan = {"table":table, "filters":filters or [], "aggregates":[{"op":"count_distinct","column":unit,"as":"total"}]}
    total = query.execute_plan(total_plan)["rows"][0][0]
    proportions = []
    for group,row in cross.iterrows():
        denominator = int(row.sum())
        for label,count in row.items():
            low,high = wilson_interval(int(count),denominator)
            proportions.append({"group":str(group),"outcome":str(label),"users":int(count),"denominator":denominator,
                                "proportion":int(count)/denominator,"ci95_wilson":[low,high]})
    test["group_proportions"] = proportions
    test["interval_scope"] = "Grup içi oranlar için bireysel %95 Wilson aralıkları; çoklu test düzeltmesi yok."
    test["scope"] = {"table":table,"dimension_table":dimension_table,"unit":unit,"dimension":dimension,"outcome":outcome,
                     "total_authors":total,"included_users":int(cross.values.sum()),"excluded_users":total-int(cross.values.sum()),
                     "period":filters or [],"note":"Analiz birimi kullanıcı/kişi ise bu anahtar seçilmelidir. Tek birime ait birden fazla gözlem veya etiket, çoklu JOIN eşleşmesi ve eksik etiketler dışlanır. Sonuç keşifseldir; seçim yanlılığı ve etiket tahmin hataları olabilir."}
    return cross,test
