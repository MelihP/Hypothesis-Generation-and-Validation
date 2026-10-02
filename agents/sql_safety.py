"""Bounded, read-only SQLite execution and strict query-plan validation."""
import math
import sqlite3
import time
from pathlib import Path
from urllib.parse import unquote

MAX_ROWS = 1000
DEFAULT_ROWS = 200
MAX_DEPTH = 3
MAX_JOINS = 3


def validate_plan(plan, depth=0):
    if depth > MAX_DEPTH or not isinstance(plan, dict):
        raise ValueError("Sorgu derinliği veya plan türü geçersiz.")
    allowed = {"table", "alias", "columns", "aggregates", "group_by", "filters", "order_by", "joins", "limit", "array_joins", "array_join"}
    if set(plan) - allowed:
        raise ValueError(f"Desteklenmeyen plan alanı: {sorted(set(plan) - allowed)}")
    if not isinstance(plan.get("table"), str) or not plan["table"]:
        raise ValueError("JSON sorgusunda zorunlu 'table' alanı eksik!")
    for field in ("columns", "aggregates", "group_by", "filters", "order_by", "joins", "array_joins"):
        if field in plan and not isinstance(plan[field], list):
            raise ValueError(f"{field} liste olmalıdır.")
        if len(plan.get(field, [])) > 100:
            raise ValueError("Plan çok fazla alan içeriyor.")
    arrays = plan.get("array_joins", plan.get("array_join", []))
    if isinstance(arrays, str):
        arrays = [arrays]
    if not isinstance(arrays, list) or len(arrays)>2:
        raise ValueError("En fazla iki ARRAY JOIN desteklenir.")
    for entry in arrays:
        if not isinstance(entry, str) and (not isinstance(entry, dict) or set(entry)-{"column", "as"} or not entry.get("column")):
            raise ValueError("ARRAY JOIN yalnızca sütun ve alias alabilir.")
    limit = plan.get("limit")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= MAX_ROWS):
        raise ValueError(f"Satır sınırı 1–{MAX_ROWS} arasında tam sayı olmalıdır.")
    if len(plan.get("joins", [])) > MAX_JOINS:
        raise ValueError("Çok fazla JOIN.")
    for join in plan.get("joins", []):
        if not isinstance(join, dict) or set(join) - {"table", "type", "alias", "on"}:
            raise ValueError("JOIN tanımı geçersiz.")
        if join.get("type", "INNER").upper() not in {"INNER", "LEFT"}:
            raise ValueError("Yalnızca INNER ve LEFT JOIN desteklenir.")
        on = join.get("on")
        conditions = on if isinstance(on, list) else [on]
        if not conditions or len(conditions) > 4:
            raise ValueError("JOIN koşulları geçersiz.")
        for condition in conditions:
            if not isinstance(condition, dict) or set(condition) != {"left", "right"}:
                raise ValueError("JOIN koşulu left/right sütun nesnesi olmalıdır; ham SQL kabul edilmez.")
    for agg in plan.get("aggregates", []):
        if not isinstance(agg, dict) or set(agg) - {"op", "column", "as"}:
            raise ValueError("Toplama tanımı geçersiz.")
        if agg.get("op") not in {"count", "count_distinct", "sum", "avg", "min", "max", "group_array"}:
            raise ValueError("Desteklenmeyen toplama operatörü.")
    for order in plan.get("order_by", []):
        if not isinstance(order, dict) or set(order) - {"column", "dir"} or order.get("dir", "asc").lower() not in {"asc", "desc"}:
            raise ValueError("Sıralama geçersiz.")
    for filt in plan.get("filters", []):
        if not isinstance(filt, dict) or set(filt) - {"column", "op", "value"} or not filt.get("column"):
            raise ValueError("Filtre tanımı geçersiz.")
        op = filt.get("op", "").upper()
        if op not in {"EQ", "NEQ", "GT", "GTE", "LT", "LTE", "LIKE", "ILIKE", "IN", "NOT_IN", "BETWEEN", "IS_NULL", "IS_NOT_NULL", "HAS", "HAS_ANY", "HAS_ALL"}:
            raise ValueError("Desteklenmeyen filtre operatörü.")
        value = filt.get("value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Sonlu sayısal değer gerekli.")
        if op == "BETWEEN" and (not isinstance(value, list) or len(value) != 2):
            raise ValueError("BETWEEN iki değer gerektirir.")
        if op in {"IN", "NOT_IN"}:
            nested = value[0] if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) else value
            if isinstance(nested, dict):
                nested = dict(nested)
                if "column" in nested and "columns" not in nested:
                    nested["columns"] = [nested.pop("column")]
                validate_plan(nested, depth + 1)
                if len(nested.get("columns", [])) != 1 or nested.get("aggregates") or nested.get("group_by"):
                    raise ValueError("IN alt sorgusu tek sütun döndürmelidir.")
            elif not isinstance(value, list) or len(value) > 100:
                raise ValueError("IN değerleri en fazla 100 elemanlı liste olmalıdır.")
        elif op in {"HAS", "HAS_ANY", "HAS_ALL"}:
            values = value if isinstance(value, list) else [value]
            if len(values) > 100 or any(isinstance(v, (list, dict)) for v in values):
                raise ValueError("Dizi filtresi en fazla 100 skaler değer alabilir.")
        elif isinstance(value, (dict, list)) and op != "BETWEEN":
            raise ValueError("Filtre değeri skaler olmalıdır.")


def database_path(db):
    name = db._engine.url.database
    if not name or name == ":memory:":
        raise ValueError("Bu yürütücü dosya tabanlı SQLite gerektirir.")
    if db._engine.url.get_backend_name() != "sqlite":
        raise ValueError("Yalnızca SQLite desteklenir.")
    return Path(unquote(name.removeprefix("file:").split("?", 1)[0])).resolve()


def read_query(path, sql, parameters=(), max_rows=MAX_ROWS, timeout=5, allowed_tables=None):
    """Even a faulty caller cannot write, attach files, or run unbounded work."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    start = time.monotonic()
    ticks = 0
    def progress():
        nonlocal ticks
        ticks += 1
        return int(time.monotonic() - start >= timeout or ticks > 20000)
    def authorize(action, first, second, database, trigger):
        if action == sqlite3.SQLITE_READ:
            return sqlite3.SQLITE_OK if allowed_tables is None or first in allowed_tables else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() in {"load_extension", "writefile", "readfile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE} else sqlite3.SQLITE_DENY
    connection.set_authorizer(authorize)
    connection.set_progress_handler(progress, 1000)
    try:
        cursor = connection.execute(sql, parameters)
        rows = cursor.fetchmany(max_rows + 1)
        return [col[0] for col in cursor.description], rows[:max_rows], len(rows) > max_rows
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc):
            raise TimeoutError("Sorgu süre veya işlem sınırını aştı.") from exc
        raise
    finally:
        connection.close()
