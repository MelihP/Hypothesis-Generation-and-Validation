"""Shared data dictionary for the SQLite dataset shipped with the application."""

DATA_CONTEXT = """
Yalnızca verilen canlı veritabanı şemasında bulunan tablo ve sütunları kullan.
Bu projenin veri sözlüğü:
- demographics: user_id, gender, age_group, user_location. Bir satır bir kullanıcıdır.
- consumer_journey: tweet_id, author_id, journey_stage, prediction_month.
- emotion_analysis: tweet_id, author_id, dominant_emotion, prediction_month.
- trending_topics: tweet_id, topic_name, prediction_month. Bir tweet birden fazla konu kaydı taşıyabilir.
- emotions_by_age_groups: age_group, dominant_emotion, volume, demographic_percentage.
İlişkiler: consumer_journey.author_id = demographics.user_id;
emotion_analysis.author_id = demographics.user_id.
Konu-yolculuk veya konu-duygu JOIN'lerinde tweet_id ve prediction_month birlikte eşleşmelidir.
Konu kayıtlarının çoğalması nedeniyle tweet sayısı için count_distinct(tweet_id) kullan;
kullanıcı sayısı için count_distinct(author_id veya user_id) kullan.
Yaş alanı age_group, yolculuk aşaması alanı journey_stage'dir.
journey_stage bir metin etiketidir; aşama filtresi EQ ile uygulanır.
Ürün, makro konu kategorisi ve bot/kurum göstergesi bu veri setinde yoktur;
bu alanları veya bunlara dayanan filtreleri uydurma. Konu analizi topic_name ile yapılabilir.
prediction_month YYYYMM biçiminde bir dönem etiketidir; olay tarihi olduğu varsayılmamalıdır.
Dönem kapsamını sorgulayarak kontrol et; mevcut olmayan dönem için veri yokluğunu belirt.
"""


def validate_query_schema(plan: dict, tables: dict[str, set[str]]) -> None:
    """Reject references that are absent or ambiguous in the live schema."""
    sources = {}
    for source in [plan, *(plan.get("joins") or [])]:
        table = source.get("table")
        if table not in tables:
            raise ValueError(f"Şemada bulunmayan tablo: {table}")
        name = source.get("alias") or table
        if name in sources:
            raise ValueError(f"Tekrarlanan tablo adı veya alias: {name}")
        sources[name] = tables[table]

    def column(name, output_aliases=()):
        if not isinstance(name, str) or not name:
            raise ValueError("Sütun adı boş olamaz.")
        if name in output_aliases:
            return
        if "." in name:
            source, col = name.split(".", 1)
            if source in sources and col in sources[source]:
                return
        elif sum(name in cols for cols in sources.values()) == 1:
            return
        raise ValueError(f"Şemada bulunmayan veya belirsiz sütun: {name}")

    for field in ("columns", "group_by"):
        for name in plan.get(field) or []:
            column(name)
    aliases = set()
    for agg in plan.get("aggregates") or []:
        name = agg.get("column")
        if name:
            column(name)
        aliases.add(agg.get("as") or f"{agg.get('op')}_{name.replace('.', '_') if name else agg.get('op')}")
    for item in plan.get("order_by") or []:
        column(item.get("column"), aliases)
    for item in plan.get("filters") or []:
        column(item.get("column"))
        value = item.get("value")
        if isinstance(value, dict) and "table" in value:
            nested = dict(value)
            if not nested.get("columns") and nested.get("column"):
                nested["columns"] = [nested.pop("column")]
            validate_query_schema(nested, tables)
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            nested = dict(value[0])
            if not nested.get("columns") and nested.get("column"):
                nested["columns"] = [nested.pop("column")]
            validate_query_schema(nested, tables)
    for join in plan.get("joins") or []:
        on = join.get("on")
        if isinstance(on, dict):
            column(on.get("left") or on.get("left_col"))
            column(on.get("right") or on.get("right_col"))
        elif isinstance(on, (list, tuple)) and len(on) == 2:
            column(on[0])
            column(on[1])
        elif isinstance(on, str) and "=" in on:
            for name in on.split("=", 1):
                column(name.strip())
        elif (join.get("type") or "INNER").upper() != "CROSS" or on:
            raise ValueError("JOIN için şemadaki iki sütunu eşleştiren koşul gerekli.")
