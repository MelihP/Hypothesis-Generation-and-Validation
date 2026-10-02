"""Five evidence-aware workflows over a shared, read-only analysis backend."""
from pathlib import Path
import os
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import get_hybrid_agent
from agents.analysis_service import AnalysisService, dataset_catalog, public_error
from agents.credentials import configure_openai_credentials
from agents.explorer import quality_report, table_plan, independent_user_comparison
from agents.query_agent import QueryAgent
from agents.results import dataframe, export_csv, export_excel
from agents.statistics import descriptive_statistics, missing_statistics

DATABASE = Path(__file__).resolve().parent / "insight_generation_bot.db"
MODES = ["🤖 Otonom İçgörü Modu", "👤 Manuel Soru Modu", "🧪 Hipotez Doğrulama Modu",
         "🔮 Tahminleme (Predictive) Modu", "📊 Veri Sorgulama ve İstatistik"]
SOURCE_OPTIONS = {"Otomatik": "auto", "Veritabanı": "sql", "Belgeler": "documents", "Veritabanı + Belgeler": "hybrid"}
st.set_page_config(page_title="Pazarlama Veri ve İçgörü Motoru", page_icon="📊", layout="wide")
st.title("📊 Pazarlama Veri ve İçgörü Motoru")
st.caption("Hesaplanmış veriler, izlenebilir kanıtlar ve kaynaklı yorumlar")


@st.cache_resource
def query_engine():
    return QueryAgent(db_uri=f"sqlite:///{DATABASE}")


@st.cache_resource
def analysis_engine(fast_model, reasoning_model):
    _, _, documents, query, rewrite, synthesis = get_hybrid_agent(
        db_uri=f"sqlite:///{DATABASE}", fast_model=fast_model, reasoning_model=reasoning_model)
    return AnalysisService(query, rewrite, synthesis, documents)


@st.cache_data(ttl=60)
def catalog_data():
    return dataset_catalog(query_engine())


def service():
    configure_openai_credentials()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Doğal dil ve yorum için OpenAI anahtarı gerekli. Tablo oluşturma ve istatistik araçları anahtarsız çalışır.")
    return analysis_engine(fast_model, reasoning_model)


def downloads(frame, key):
    left, right = st.columns(2)
    left.download_button("CSV indir", export_csv(frame), f"{key}.csv", "text/csv", key=f"csv_{key}")
    right.download_button("Excel indir", export_excel(frame), f"{key}.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"excel_{key}")


def render_result(result, key):
    st.caption(result.get("question", ""))
    if result.get("status") == "failed":
        st.error(result.get("error", "Sorgu başarısız."))
        return
    for warning in result.get("warnings", []):
        st.warning(warning)
    if result.get("status") == "empty":
        st.info("Sonuç boş veya sayım sıfır. Bu durum tek başına iş performansı hakkında kanıt değildir.")
    frame = dataframe(result)
    st.dataframe(frame, hide_index=True, width="stretch")
    if not frame.empty:
        downloads(frame, key)
    if result.get("source_type") == "document":
        return
    metadata = result.get("metadata", [])
    metrics = [m["name"] for m in metadata if m["role"] == "metric" and m["type"] == "number"]
    dimensions = [m["name"] for m in metadata if m["role"] in {"dimension", "period"}]
    if metrics and dimensions and not frame.empty:
        metric = st.selectbox("Grafik metriği", metrics, key=f"metric_{key}")
        dimension = st.selectbox("Grafik kırılımı", dimensions, key=f"dimension_{key}")
        unit = next(m["unit"] for m in metadata if m["name"] == metric)
        if dimension == "prediction_month":
            fig = px.line(frame.sort_values(dimension), x=dimension, y=metric, markers=True)
            fig.update_xaxes(type="category")
        else:
            fig = px.bar(frame, x=dimension, y=metric,
                         color=dimensions[1] if len(dimensions)>1 and dimensions[1]!=dimension else None,
                         barmode="group")
        fig.update_layout(yaxis_title=f"{metric} ({unit})")
        st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
        counts = [m["name"] for m in metadata if m.get("operation") in {"count", "count_distinct"}]
        if metric in counts and st.checkbox("Görünen tablo paylarını göster", key=f"shares_{key}"):
            total = frame[metric].sum()
            shares = frame.copy()
            shares["Görünen toplam payı (%)"] = shares[metric]/total*100 if total else None
            st.caption("Payda yalnızca gösterilen grup hacimlerinin toplamıdır; nüfus payı değildir. Çok etiketli veya kesilmiş sonuçlar tam dağılımı temsil etmez.")
            st.dataframe(shares, hide_index=True, width="stretch")
    if len(dimensions)>=2 and metrics and not frame.empty:
        with st.expander("Çapraz tablo"):
            cross = frame.pivot_table(index=dimensions[0], columns=dimensions[1], values=metrics[0], aggfunc="sum", fill_value=0)
            st.dataframe(cross, width="stretch")
    with st.expander("İstatistik ve veri kalitesi"):
        st.caption("Bu özet gösterilen sonuç satırlarına aittir; gruplanmış sayıların ortalaması ham gözlem ortalaması değildir.")
        summary = descriptive_statistics(frame, metadata)
        if not summary.empty:
            st.dataframe(summary, hide_index=True, width="stretch")
        st.dataframe(missing_statistics(frame), hide_index=True, width="stretch")
    with st.expander("Analiz kapsamı ve sorgu ayrıntıları"):
        st.json({"grain": result.get("grain"), "filters": result.get("filters"), "metadata": metadata,
                 "row_count": result.get("row_count"), "truncated": result.get("truncated")})
        st.code(result.get("sql", ""), language="sql")
        st.json(result.get("parameters", []))


def render_answer(answer, key):
    if answer.get("error"):
        st.error(answer["error"])
    if answer.get("status") == "partial":
        st.warning("Kısmi sonuç: eksik adımlar var; yorum yalnızca başarılı kanıtlarla sınırlıdır.")
    if answer.get("hypotheses"):
        st.json(answer["hypotheses"])
        st.caption("Bu bölüm betimsel kanıt değerlendirmesidir. Hesaplanmış test için Veri Sorgulama ve İstatistik modunu kullanın.")
    if answer.get("insight"):
        st.markdown(answer["insight"])
    for i, result in enumerate(answer.get("results", [])):
        with st.expander(f"Kanıt {i+1}: {result.get('status', '')}", expanded=True):
            render_result(result, f"{key}_{i}")


profile = st.sidebar.selectbox("Model profili", ["Hibrit", "Tasarruf", "Hassasiyet"])
fast_model = "gpt-4o" if profile == "Hassasiyet" else "gpt-4o-mini"
reasoning_model = "gpt-4o-mini" if profile == "Tasarruf" else "gpt-4o"
mode = st.sidebar.radio("Çalışma modu", MODES)
source_name = st.sidebar.selectbox("Analiz kaynağı", list(SOURCE_OPTIONS))
source = SOURCE_OPTIONS[source_name]
st.sidebar.caption("SQL salt okunur; en fazla 1.000 sonuç satırı. Belgeler için Pinecone bağlantısı gerekir.")
try:
    catalog = catalog_data()
except Exception as exc:
    st.error(public_error(exc))
    st.stop()
periods = sorted({p for item in catalog.values() for p in item["periods"]})
st.caption("Veri dönemleri: " + (", ".join(str(p) for p in periods) if periods else "Dönem bilgisi yok"))
if len(periods)<2:
    st.info("Mevcut veride çok dönemli trend analizi yapılamıyor. Eksik dönem, sıfır gözlem anlamına gelmez.")
with st.sidebar.expander("Veri kataloğu"):
    st.dataframe(pd.DataFrame([{"Tablo": table, "Kayıt": info["row_count"], "Dönem": ", ".join(map(str, info["periods"]))}
                              for table, info in catalog.items()]), hide_index=True)

st.session_state.setdefault("answers", {})
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("pending_clarification", None)

if mode == MODES[4]:
    st.subheader("Veri Sorgulama ve İstatistik")
    kind = st.radio("İşlem", ["Tablo oluştur", "Doğal dilde sorgula", "Veri kalitesi", "İstatistiksel karşılaştırma"], horizontal=True)
    query = query_engine()
    if kind == "Doğal dilde sorgula":
        question = st.text_input("Veri sorusu", placeholder="Yaş gruplarına göre kullanıcı sayısını çıkar")
        st.caption("Sonuç SQL/Python ile hesaplanır; yönetici yorumu üretilmez.")
        if st.button("Veriyi sorgula") and question.strip():
            try:
                planner = service().query
                plan = planner.generate_query_json(question)
                st.session_state.answers["explorer"] = query.execute_plan(plan, question)
            except Exception as exc:
                st.session_state.answers.pop("explorer", None)
                st.error(public_error(exc))
        if "explorer" in st.session_state.answers:
            render_result(st.session_state.answers["explorer"], "natural")
    elif kind == "Tablo oluştur":
        table = st.selectbox("Tablo", list(catalog))
        info = catalog[table]
        groups = st.multiselect("Gruplama sütunları (çapraz tablo için iki sütun)", info["columns"], max_selections=2)
        metrics = {"Kayıt sayısı": "records"}
        if "tweet_id" in info["columns"]:
            metrics["Farklı tweet sayısı"] = "tweets"
        if "user_id" in info["columns"] or "author_id" in info["columns"]:
            metrics["Farklı kullanıcı sayısı"] = "users"
        if "volume" in info["columns"]:
            metrics["Toplam hacim"] = "volume"
        metric = metrics[st.selectbox("Metrik", list(metrics))]
        period = st.selectbox("Dönem filtresi", ["Tümü", *info["periods"]])
        limit = st.number_input("Satır sınırı", min_value=1, max_value=1000, value=200)
        filter_column = st.selectbox("Ek filtre sütunu", ["Yok", *info["columns"]])
        filter_value = st.text_input("Filtre değeri (tam eşleşme)") if filter_column != "Yok" else ""
        if st.button("Tabloyu oluştur"):
            try:
                filters = []
                if period != "Tümü":
                    filters.append({"column": "prediction_month", "op": "EQ", "value": period})
                if filter_column != "Yok":
                    filters.append({"column": filter_column, "op": "EQ", "value": filter_value})
                result = query.execute_plan(table_plan(table, groups, metric, filters, int(limit)), f"{table}: {metric}")
                st.session_state.answers["builder"] = result
            except Exception as exc:
                st.session_state.answers.pop("builder", None)
                st.error(public_error(exc))
        if "builder" in st.session_state.answers:
            render_result(st.session_state.answers["builder"], "builder")
    elif kind == "Veri kalitesi":
        table = st.selectbox("Kalite kontrolü tablosu", list(catalog))
        if st.button("Eksik veri analizini çalıştır"):
            try:
                st.session_state.answers["quality"] = quality_report(query, table)
            except Exception as exc:
                st.error(public_error(exc))
        if "quality" in st.session_state.answers:
            frame = st.session_state.answers["quality"]
            st.caption("Tam tablo üzerinde NULL değer kontrolü; boş metinler NULL sayılmaz.")
            st.dataframe(frame, hide_index=True, width="stretch")
            downloads(frame, "quality")
    else:
        st.caption("Yaş/cinsiyet ile duygu/yolculuk etiketi arasındaki ilişki. Her kullanıcı tek gözlemle temsil edilir; tekrarlı kayıtlar dışlanır.")
        table = st.selectbox("Karşılaştırma verisi", ["emotion_analysis", "consumer_journey"])
        dimension = st.selectbox("Demografik kırılım", ["age_group", "gender"])
        period = st.selectbox("Karşılaştırma dönemi", catalog[table]["periods"] or [None])
        if st.button("İstatistiksel testi çalıştır"):
            try:
                cross, test = independent_user_comparison(query, table, dimension, period)
                st.session_state.answers["statistical"] = (cross, test)
            except Exception as exc:
                st.session_state.answers.pop("statistical", None)
                st.error(public_error(exc))
        if "statistical" in st.session_state.answers:
            cross, test = st.session_state.answers["statistical"]
            st.dataframe(cross, width="stretch")
            if test["status"] == "success":
                st.caption(test["method"])
                left, middle, right = st.columns(3)
                left.metric("p-değeri", f"{test['p_value']:.6g}")
                effect = test.get("cramers_v_corrected")
                middle.metric("Cramér V (düzeltilmiş)", f"{effect:.4f}" if effect is not None else "—")
                right.metric("Analize alınan kullanıcı", test["n"])
                st.info(test["interpretation"] + ". " + test["warning"])
                if "proportion_difference" in test:
                    low, high = test["difference_ci95"]
                    st.write(f"İlk sonuç sütunu için iki grup arasındaki oran farkı: {test['proportion_difference']*100:.2f} yüzde puan. %95 aralık: [{low*100:.2f}, {high*100:.2f}].")
            else:
                st.warning(test["reason"])
            scope = test["scope"]
            st.caption(f"Dönem: {scope['period']} · Toplam yazar: {scope['total_authors']} · Dahil: {scope['included_users']} · Hariç: {scope['excluded_users']}")
            st.caption(scope["note"])
            if test.get("group_proportions"):
                proportions = pd.DataFrame([{"Grup": p["group"], "Etiket": p["outcome"], "Kullanıcı": p["users"],
                    "Grup toplamı": p["denominator"], "Oran (%)": p["proportion"]*100,
                    "%95 alt sınır": p["ci95_wilson"][0]*100, "%95 üst sınır": p["ci95_wilson"][1]*100}
                    for p in test["group_proportions"]])
                st.dataframe(proportions, hide_index=True, width="stretch")
                st.caption(test["interval_scope"])
                downloads(proportions, "proportion_intervals")
            st.caption("Beklenen hücre sayıları düşükse 2×2 tabloda Fisher testi kullanılır; daha büyük tabloda sonuç yetersiz sayılır. Test keşifseldir ve nedensellik kanıtlamaz.")
            downloads(cross.reset_index(), "comparison")

elif mode == MODES[1]:
    st.subheader("Soru ve takip analizi")
    if st.sidebar.button("Konuyu sıfırla"):
        st.session_state.chat_history = []
        st.session_state.pending_clarification = None
        st.rerun()
    for i, turn in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(turn["raw_question"])
        with st.chat_message("assistant"):
            render_answer(turn["answer"], f"turn_{i}")
    pending = st.session_state.pending_clarification
    request = None
    if pending:
        st.warning(pending["message"])
        for i, option in enumerate(pending["options"]):
            if st.button(option["label"], key=f"clarify_{i}"):
                request = {**pending, "resolved": f"{pending['resolved']} ({option['context']})"}
                st.session_state.pending_clarification = None
    user_input = st.chat_input("Veri, belge veya takip sorunuzu yazın")
    if user_input:
        try:
            engine = service()
            history = [{"user": t["raw_question"], "resolved_query": t["answer"]["resolved_query"],
                        "context": t["answer"].get("context", []), "assistant_summary": t["answer"].get("insight", "")}
                       for t in st.session_state.chat_history if t["answer"]["status"] in {"success", "partial"}]
            resolved = engine.rewrite.contextualize_query(user_input, history)
            clarification = engine.rewrite.assess_clarification_need(resolved, engine.query.schema)
            request = {"raw": user_input, "resolved": resolved, "source": source}
            if clarification.get("needs_clarification") and clarification.get("options"):
                st.session_state.pending_clarification = {**request, "message": clarification.get("clarification_message", "Analiz odağını seçin"), "options": clarification["options"][:4]}
                request = None
                st.rerun()
        except Exception as exc:
            st.error(public_error(exc))
    if request:
        with st.spinner("Kanıtlar toplanıyor..."):
            try:
                answer = service().analyze(request["resolved"], source=request["source"])
                st.session_state.chat_history.append({"raw_question": request["raw"], "answer": answer})
                st.session_state.chat_history = st.session_state.chat_history[-20:]
                st.rerun()
            except Exception as exc:
                st.error(public_error(exc))
else:
    internal_mode = {MODES[0]: "autonomous", MODES[2]: "hypothesis", MODES[3]: "predictive"}[mode]
    st.subheader(mode)
    question = ""
    if internal_mode != "autonomous":
        question = st.text_input("Hipotez veya araştırma sorusu" if internal_mode=="hypothesis" else "Trend sorusu")
    if st.button("Analizi başlat"):
        try:
            with st.spinner("Analiz hazırlanıyor..."):
                engine = service()
                if internal_mode == "autonomous":
                    question = engine.rewrite.generate_macro_question(engine.query.schema)
                if not question.strip():
                    raise ValueError("Bir soru girin.")
                answer = engine.analyze(question, mode=internal_mode, source=source)
                st.session_state.answers[internal_mode] = answer
        except Exception as exc:
            st.session_state.answers.pop(internal_mode, None)
            st.error(public_error(exc))
    if internal_mode in st.session_state.answers:
        render_answer(st.session_state.answers[internal_mode], internal_mode)
