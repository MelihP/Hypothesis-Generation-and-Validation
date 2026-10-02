"""Live-schema validation for both backends, including ARRAY JOIN aliases."""


def validate_query_schema(plan, catalog):
    sources = {}
    for source in [plan, *plan.get("joins", [])]:
        table = source.get("table")
        if table not in catalog:
            raise ValueError(f"Şemada bulunmayan tablo: {table}")
        alias = source.get("alias") or table
        if alias in sources:
            raise ValueError(f"Tekrarlanan tablo veya alias: {alias}")
        sources[alias] = catalog[table]["types"]
    virtual = {}

    def column(name, outputs=()):
        if not isinstance(name, str) or not name:
            raise ValueError("Sütun adı boş olamaz.")
        if name in outputs:
            return "output"
        if name in virtual:
            return virtual[name]
        if "." in name:
            source, field = name.split(".", 1)
            if source in sources and field in sources[source]:
                return sources[source][field]
        else:
            matches = [types[name] for types in sources.values() if name in types]
            if len(matches)==1:
                return matches[0]
        raise ValueError(f"Şemada bulunmayan veya belirsiz sütun: {name}")

    arrays = plan.get("array_joins", plan.get("array_join", []))
    arrays = [arrays] if isinstance(arrays, str) else arrays
    for entry in arrays:
        name = entry if isinstance(entry, str) else entry["column"]
        dtype = column(name)
        if not dtype.startswith("Array("):
            raise ValueError("ARRAY JOIN bir Array sütunu gerektirir.")
        if isinstance(entry, dict) and entry.get("as"):
            if entry["as"] in virtual or any(entry["as"] in types for types in sources.values()):
                raise ValueError("ARRAY JOIN alias mevcut sütunla çakışıyor.")
            virtual[entry["as"]] = dtype[6:-1]
    for field in ("columns", "group_by"):
        for name in plan.get(field, []):
            column(name)
    outputs = set()
    for agg in plan.get("aggregates", []):
        name = agg.get("column")
        if name:
            column(name)
        outputs.add(agg.get("as") or f"{agg['op']}_{str(name or agg['op']).replace('.', '_')}")
    for order in plan.get("order_by", []):
        column(order.get("column"), outputs)
    for join in plan.get("joins", []):
        conditions = join["on"] if isinstance(join["on"], list) else [join["on"]]
        for condition in conditions:
            column(condition["left"])
            column(condition["right"])
    for filt in plan.get("filters", []):
        dtype = column(filt["column"])
        if filt["op"].upper() in {"HAS", "HAS_ANY", "HAS_ALL"} and any(t.startswith("Array(") for types in sources.values() for t in types.values()) and not dtype.startswith("Array("):
            raise ValueError("HAS filtresi bir Array sütunu gerektirir.")
        value = filt.get("value")
        nested = value[0] if isinstance(value, list) and len(value)==1 and isinstance(value[0],dict) else value
        if isinstance(nested, dict):
            nested = dict(nested)
            if "column" in nested and "columns" not in nested:
                nested["columns"] = [nested.pop("column")]
            validate_query_schema(nested, catalog)
