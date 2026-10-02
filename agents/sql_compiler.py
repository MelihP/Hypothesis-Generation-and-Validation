"""
Deterministik JSON-to-SQL Derleyicisi (Deterministic JSON-to-SQL Compiler)
Yapılandırılmış JSON sorgu nesnelerini (SPJQ) güvenli SQL ifadelerine dönüştürür.
Desteklenen lehçeler (dialects): 'sqlite', 'clickhouse'
"""

import re
from typing import Any, Optional
from agents.sql_safety import validate_plan

# --- 1. OPERATÖR HARİTASI VE TOPLAMA FONKSİYONLARI ---
_FILTER_OP_TO_SQL = {
    "EQ": "=",
    "NEQ": "!=",
    "GT": ">",
    "GTE": ">=",
    "LT": "<",
    "LTE": "<=",
    "LIKE": "LIKE",
    "ILIKE": "ILIKE",
    "IN": "IN",
    "NOT_IN": "NOT IN",
    "IS_NULL": "IS NULL",
    "IS_NOT_NULL": "IS NOT NULL",
    "BETWEEN": "BETWEEN",
    "HAS": "HAS",
    "HAS_ANY": "HAS_ANY",
    "HAS_ALL": "HAS_ALL",
}

_AGG_OPS = frozenset({"count", "count_distinct", "sum", "avg", "min", "max", "group_array"})
_DANGEROUS_IDENT_RE = re.compile(r"[;\x00]|--|/\*")


# --- 2. GÜVENLİK VE KAÇIŞLAMA YARDIMCILARI ---

def quote_ident(name: str, dialect: str = "sqlite", quote_char: Optional[str] = None) -> str:
    """
    Tablo ve sütun adlarını lehçeye uygun kaçış karakteriyle tırnak içine alır.
    SQLite için çift tırnak ("), ClickHouse için backtick (`) kullanılır.
    """
    if not name or _DANGEROUS_IDENT_RE.search(name):
        raise ValueError(f"Geçersiz tanımlayıcı (Identifier): {name!r}")

    dialect = (dialect or "sqlite").lower().strip()
    if quote_char is None:
        quote_char = "`" if dialect == "clickhouse" else '"'

    # json_each(tablo.sutun) gibi SQLite fonksiyonlarını koru
    m = re.match(r"^json_each\((.+)\)$", name.strip(), re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        return f"json_each({quote_ident(inner, dialect=dialect, quote_char=quote_char)})"

    # tablo.sutun formatındaki ilişkisel tanımlayıcıları ayrı ayrı tırnakla
    if "." in name:
        return ".".join(quote_ident(p.strip(), dialect=dialect, quote_char=quote_char) for p in name.split("."))

    escaped = name.replace(quote_char, quote_char * 2)
    return f"{quote_char}{escaped}{quote_char}"


def _lit(v: Any) -> str:
    """Değerleri SQL injection güvenliği için kaçışlayarak biçimlendirir."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("\x00", "").replace("'", "''")
    return f"'{s}'"


def _contains_lit(v: Any) -> str:
    """LIKE / ILIKE sorguları için güvenli '%metin%' formatına dönüştürür."""
    s = str(v).replace("\x00", "").replace("'", "''")
    return f"'%{s}%'"


def _format_array_lit(vals: list[Any]) -> str:
    """ClickHouse array formatına dönüştürür (['val1', 'val2'])."""
    return f"[{', '.join(literal(v) for v in vals)}]"


def _compile_join(j: dict[str, Any], dialect: str = "sqlite") -> str:
    on = j["on"]
    conditions = on if isinstance(on, list) else [on]
    if not conditions or any(not isinstance(c, dict) or set(c)!={"left", "right"} for c in conditions):
        raise ValueError("Ham JOIN koşulu kabul edilmez.")
    table = quote_ident(j["table"], dialect=dialect)
    if j.get("alias"):
        table += f" AS {quote_ident(j['alias'], dialect=dialect)}"
    clause = " AND ".join(f"{quote_ident(c['left'],dialect=dialect)} = {quote_ident(c['right'],dialect=dialect)}" for c in conditions)
    return f"{j.get('type', 'INNER').upper()} JOIN {table} ON {clause}"


# --- 3. JSON-TO-SQL DERLEYİCİSİ (SQLITE & CLICKHOUSE) ---

def compile_json_to_sql(query_json: dict[str, Any], dialect: str = "sqlite", parameters: Optional[list] = None) -> str:
    """
    Yapılandırılmış JSON sorgu nesnesini deterministik ve güvenli SQL ifadesine dönüştürür.
    Desteklenen lehçeler: 'sqlite', 'clickhouse'
    """
    dialect = (dialect or "sqlite").lower().strip()
    validate_plan(query_json)
    if dialect not in {"sqlite", "clickhouse"}:
        raise ValueError("Desteklenmeyen SQL lehçesi.")
    def literal(value):
        if parameters is None:
            return _lit(value)
        index = len(parameters)
        parameters.append(value)
        return f"%(p{index})s" if dialect == "clickhouse" else "?"
    def contains(value):
        return literal("%"+str(value)+"%")
    def array_literal(values):
        return "["+", ".join(literal(v) for v in values)+"]"
    table = query_json.get("table")
    if not table:
        raise ValueError("JSON sorgusunda zorunlu 'table' alanı eksik!")

    q_table = quote_ident(table, dialect=dialect)
    alias = query_json.get("alias")
    if alias:
        q_table += f" AS {quote_ident(str(alias), dialect=dialect)}"

    joins = query_json.get("joins") or []
    columns = query_json.get("columns") or []
    aggregates = query_json.get("aggregates") or []
    group_by = query_json.get("group_by") or []
    filters = query_json.get("filters") or []
    order_by = query_json.get("order_by") or []
    limit = query_json.get("limit")
    array_joins = query_json.get("array_joins") or query_json.get("array_join") or []

    select_parts: list[str] = [quote_ident(str(g), dialect=dialect) for g in group_by]

    for agg in aggregates:
        if not isinstance(agg, dict):
            continue
        op = (agg.get("op") or "").lower().strip()
        if op not in _AGG_OPS:
            raise ValueError(f"Desteklenmeyen toplama operatörü: {op!r}")
        col = agg.get("column")
        safe_col_name = str(col).replace(".", "_") if col else op
        alias_agg = agg.get("as") or f"{op}_{safe_col_name}"
        q_alias = quote_ident(alias_agg, dialect=dialect)

        if op == "count" and not col:
            expr = "count(*)"
        elif op == "count_distinct":
            if not col:
                raise ValueError("count_distinct işlemi için sütun adı zorunludur.")
            q_col = quote_ident(str(col), dialect=dialect)
            expr = f"count(DISTINCT {q_col})"
        elif op == "count":
            expr = f"count({quote_ident(str(col), dialect=dialect)})"
        elif op == "group_array":
            q_col = quote_ident(str(col), dialect=dialect)
            if dialect == "clickhouse":
                expr = f"groupArray({q_col})"
            else:
                expr = f"group_concat({q_col})"
        else:
            if not col:
                raise ValueError(f"'{op}' toplama işlemi için sütun adı zorunludur.")
            expr = f"{op}({quote_ident(str(col), dialect=dialect)})"

        select_parts.append(f"{expr} AS {q_alias}")

    if not select_parts:
        select_parts = [quote_ident(str(c), dialect=dialect) for c in columns] if columns else ["*"]

    select_clause = ", ".join(select_parts)
    from_clause = q_table

    sql = f"SELECT {select_clause} FROM {from_clause}"
    for j in joins:
        sql += " " + _compile_join(j, dialect=dialect)
    if array_joins:
        if dialect != "clickhouse":
            raise ValueError("ARRAY JOIN yalnızca ClickHouse üzerinde desteklenir.")
        aj_list = array_joins if isinstance(array_joins, list) else [array_joins]
        expressions = []
        for aj in aj_list:
            if isinstance(aj, str):
                expressions.append(quote_ident(aj, dialect=dialect))
            else:
                expression = quote_ident(aj["column"], dialect=dialect)
                if aj.get("as"):
                    expression += " AS " + quote_ident(aj["as"], dialect=dialect)
                expressions.append(expression)
        sql += " ARRAY JOIN " + ", ".join(expressions)

    # WHERE koşullarını derle
    where_parts: list[str] = []
    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op_key = (f.get("op") or "").upper().strip()
        sql_op = _FILTER_OP_TO_SQL.get(op_key)
        if not col or sql_op is None:
            continue
        q_col = quote_ident(str(col), dialect=dialect)
        val = f.get("value")

        if sql_op in ("IS NULL", "IS NOT NULL"):
            where_parts.append(f"{q_col} {sql_op}")
        elif sql_op in ("IN", "NOT IN"):
            # Nested subquery kontrolü
            if isinstance(val, dict) and "table" in val:
                subquery_dict = dict(val)
                if not subquery_dict.get("columns") and subquery_dict.get("column"):
                    subquery_dict["columns"] = [subquery_dict.pop("column")]
                sub_sql = compile_json_to_sql(subquery_dict, dialect=dialect, parameters=parameters)
                where_parts.append(f"{q_col} {sql_op} ({sub_sql})")
            elif isinstance(val, (list, tuple)) and len(val) == 1 and isinstance(val[0], dict) and "table" in val[0]:
                subquery_dict = dict(val[0])
                if not subquery_dict.get("columns") and subquery_dict.get("column"):
                    subquery_dict["columns"] = [subquery_dict.pop("column")]
                sub_sql = compile_json_to_sql(subquery_dict, dialect=dialect, parameters=parameters)
                where_parts.append(f"{q_col} {sql_op} ({sub_sql})")
            else:
                vals = val if isinstance(val, (list, tuple)) else [val]
                if not vals:
                    where_parts.append("0 = 1" if sql_op == "IN" else "1 = 1")
                else:
                    where_parts.append(f"{q_col} {sql_op} ({', '.join(literal(v) for v in vals)})")
        elif sql_op == "BETWEEN":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                where_parts.append(f"{q_col} BETWEEN {literal(val[0])} AND {literal(val[1])}")
        elif op_key == "HAS":
            if dialect == "clickhouse":
                if isinstance(val, (list, tuple)):
                    where_parts.append(f"hasAny({q_col}, {array_literal(val)})")
                else:
                    where_parts.append(f"has({q_col}, {literal(val)})")
            else:
                where_parts.append(f"{q_col} LIKE {contains(val)}")
        elif op_key == "HAS_ANY":
            vals = val if isinstance(val, (list, tuple)) else [val]
            if dialect == "clickhouse":
                where_parts.append(f"hasAny({q_col}, {array_literal(vals)})")
            else:
                conditions = [f"{q_col} LIKE {contains(v)}" for v in vals]
                where_parts.append(f"({' OR '.join(conditions)})" if conditions else "0 = 1")
        elif op_key == "HAS_ALL":
            vals = val if isinstance(val, (list, tuple)) else [val]
            if dialect == "clickhouse":
                where_parts.append(f"hasAll({q_col}, {array_literal(vals)})")
            else:
                conditions = [f"{q_col} LIKE {contains(v)}" for v in vals]
                where_parts.append(f"({' AND '.join(conditions)})" if conditions else "1 = 1")
        elif op_key == "ILIKE":
            if dialect == "clickhouse":
                where_parts.append(f"{q_col} ILIKE {contains(val)}")
            else:
                where_parts.append(f"{q_col} LIKE {contains(val)}")
        elif op_key == "LIKE":
            where_parts.append(f"{q_col} LIKE {contains(val)}")
        elif val is None and op_key in ("EQ", "NEQ"):
            where_parts.append(f"{q_col} IS {'NOT ' if op_key == 'NEQ' else ''}NULL")
        else:
            where_parts.append(f"{q_col} {sql_op} {literal(val)}")

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    if group_by:
        sql += " GROUP BY " + ", ".join(quote_ident(str(g), dialect=dialect) for g in group_by)

    order_parts: list[str] = []
    for o in order_by:
        if not isinstance(o, dict):
            continue
        col = o.get("column")
        if not col:
            continue
        direction = "DESC" if str(o.get("dir", "")).lower() == "desc" else "ASC"
        order_parts.append(f"{quote_ident(str(col), dialect=dialect)} {direction}")

    if order_parts:
        sql += " ORDER BY " + ", ".join(order_parts)

    if limit is not None:
        try:
            sql += f" LIMIT {int(limit)}"
        except (ValueError, TypeError):
            pass

    return sql
