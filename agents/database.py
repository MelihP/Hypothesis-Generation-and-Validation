"""Backend-neutral read-only access; native ClickHouse arrays retain their types."""
from pathlib import Path
import re
from time import monotonic

from sqlalchemy import inspect
from langchain_community.utilities import SQLDatabase

from agents.sql_safety import read_query, database_path


def quote(name, dialect):
    from agents.sql_compiler import quote_ident
    return quote_ident(name, dialect=dialect)


class SQLiteBackend:
    dialect = "sqlite"
    fallback = False

    def __init__(self, path=None, db=None, fallback=False):
        path = Path(path).resolve() if path else None
        if db is None and (path is None or not path.is_file()):
            raise ValueError("SQLite veritabanı bulunamadı.")
        self.db = db or SQLDatabase.from_uri(f"sqlite:///file:{path}?mode=ro&uri=true", sample_rows_in_table_info=0)
        self.fallback = fallback
        self.path = database_path(self.db)

    def get_table_info(self):
        return self.db.get_table_info()

    def read(self, sql, parameters=(), max_rows=1000, timeout=5):
        return read_query(self.path, sql, parameters, max_rows, timeout,
                          allowed_tables=set(self.db.get_usable_table_names()))

    def catalog(self):
        inspector = inspect(self.db._engine)
        output = {}
        for table in self.db.get_usable_table_names():
            cols = inspector.get_columns(table)
            types = {col["name"]: str(col["type"]) for col in cols}
            _, rows, _ = self.read(f"SELECT count(*) FROM {quote(table, self.dialect)}")
            periods = []
            if "prediction_month" in types:
                _, data, cut = self.read(f"SELECT DISTINCT prediction_month FROM {quote(table, self.dialect)} WHERE prediction_month IS NOT NULL ORDER BY prediction_month LIMIT 1001")
                if not cut:
                    periods = [row[0] for row in data]
            output[table] = {"columns": list(types), "types": types, "row_count": rows[0][0], "periods": periods,
                             "period_column": "prediction_month" if "prediction_month" in types else None,
                             "period_expression": quote("prediction_month", self.dialect) if "prediction_month" in types else None}
        return output


class ClickHouseBackend:
    dialect = "clickhouse"
    fallback = False
    core_tables = ("tweet_predictions", "tweets", "users", "user_factors")

    def __init__(self, client, database="default"):
        self.client = client
        self.database = database
        self._catalog = None
        self._catalog_at = 0

    def read(self, sql, parameters=(), max_rows=1000, timeout=5):
        if not re.match(r"^\s*(SELECT|WITH|DESCRIBE)\b", sql, re.IGNORECASE) or ";" in sql:
            raise ValueError("Yalnızca okuma sorguları kabul edilir.")
        # readonly forbids mutations even if a future caller passes malformed SQL.
        settings = {"readonly": 1, "max_execution_time": timeout, "max_result_rows": max_rows+1,
                    "result_overflow_mode": "throw", "max_memory_usage": 268435456,
                    "max_rows_to_read": 10000000, "read_overflow_mode": "throw"}
        params = {f"p{i}": value for i, value in enumerate(parameters)} if isinstance(parameters, (tuple, list)) else parameters
        result = self.client.query(sql, parameters=params or None, settings=settings)
        rows = result.result_rows
        return list(result.column_names), rows[:max_rows], len(rows)>max_rows

    def catalog(self):
        if self._catalog is not None and monotonic() - self._catalog_at < 60:
            return self._catalog
        _, tables, _ = self.read("SELECT name, total_rows FROM system.tables WHERE database = %(p0)s AND name IN %(p1)s",
                                 [self.database, list(self.core_tables)])
        output = {}
        for table, count in tables:
            _, columns, _ = self.read(f"DESCRIBE TABLE {quote(table, self.dialect)}")
            types = {row[0]: row[1] for row in columns}
            period = next((name for name in ("prediction_month", "created_at", "tweet_date", "date", "timestamp", "updated_at")
                           if name in types and (name == "prediction_month" or "Date" in types[name])), None)
            expression = quote(period, self.dialect) if period else None
            if period and period != "prediction_month":
                expression = f"toYYYYMM({expression})"
            output[table] = {"columns": list(types), "types": types, "row_count": count,
                             "periods": [], "period_column": period, "period_expression": expression}
        if not output:
            raise ValueError("ClickHouse bağlantısında izin verilen çekirdek tablolar bulunamadı.")
        self._catalog = output
        self._catalog_at = monotonic()
        return output

    def periods(self, table):
        info = self.catalog()[table]
        if not info["period_expression"]:
            return []
        expr = info["period_expression"]
        _, rows, cut = self.read(f"SELECT DISTINCT {expr} AS period FROM {quote(table, self.dialect)} WHERE {quote(info['period_column'], self.dialect)} IS NOT NULL ORDER BY period LIMIT 1001")
        if cut:
            raise ValueError("Dönem listesi kesildi; analiz kapsamını daraltın.")
        return [row[0] for row in rows]

    def get_table_info(self):
        return "\n".join(f"CREATE TABLE {quote(table, self.dialect)} (" +
                          ", ".join(f"{quote(name, self.dialect)} {dtype}" for name,dtype in info["types"].items()) + ");"
                          for table, info in self.catalog().items())


def adapt_database(db):
    if isinstance(db, (SQLiteBackend, ClickHouseBackend)):
        return db
    if isinstance(db, SQLDatabase):
        return SQLiteBackend(db=db)
    return None
