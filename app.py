import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re

# Dinamik motor fonksiyonu
from agent import get_hybrid_agent

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Enlighty AI - Strategic Marketing Insight Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Destekli Pazarlama İçgörü Motoru")
st.caption("Stratejik Kök Neden Analizi, Demografik Sentez ve Hipotez Doğrulama Platformu")


# --- 2. SOL MENÜ: DİNAMİK MODEL SEÇİM PANELİ (TIERED ROUTING) ---
st.sidebar.subheader("⚙️ Model Yapılandırması")

profile_choice = st.sidebar.selectbox(
    "Çalışma Profili Seçin:",
    [
        "⚡ Hibrit Mod (Önerilen)",
        "🧠 Maksimum Hassasiyet",
        "💸 Maksimum Tasarruf & Hız",
        "🛠️ Özel Yapılandırma (Custom)"
    ],
    help="Sistemin sorgu planlama ve stratejik sentez adımlarında kullanacağı modelleri belirler."
)

if profile_choice == "⚡ Hibrit Mod (Önerilen)":
    selected_fast_model = "gpt-4o-mini"
    selected_reasoning_model = "gpt-4o"
    profile_badge = "⚡ Hibrit (Mini + GPT-4o)"
    profile_desc = "SQL/Sorgu motoru için hızlı `gpt-4o-mini`, stratejik yönetici sentezi için derin `gpt-4o` devrede."
elif profile_choice == "🧠 Maksimum Hassasiyet":
    selected_fast_model = "gpt-4o"
    selected_reasoning_model = "gpt-4o"
    profile_badge = "🧠 Full GPT-4o"
    profile_desc = "Tüm aşamalarda en güçlü akıl yürütme modeli kullanılır. Maliyeti daha yüksektir."
elif profile_choice == "💸 Maksimum Tasarruf & Hız":
    selected_fast_model = "gpt-4o-mini"
    selected_reasoning_model = "gpt-4o-mini"
    profile_badge = "💸 Full GPT-4o-mini"
    profile_desc = "Tüm süreçler `gpt-4o-mini` ile çalışır. Ultra hızlı ve %90 daha düşük API maliyeti sağlar."
else:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        selected_fast_model = st.selectbox("1. Kademe (SQL):", ["gpt-4o-mini", "gpt-4o"])
    with c2:
        selected_reasoning_model = st.selectbox("2. Kademe (Sentez):", ["gpt-4o", "gpt-4o-mini"])
    profile_badge = f"🛠️ Özel ({selected_fast_model} + {selected_reasoning_model})"
    profile_desc = "Kullanıcı tanımlı model eşleştirmesi."

st.sidebar.caption(profile_desc)

# Seçilen modele göre motoru önbellekten yükleme
@st.cache_resource(show_spinner=False)
def load_app_engine(f_model: str, r_model: str):
    return get_hybrid_agent(fast_model=f_model, reasoning_model=r_model)

engine_bundle = load_app_engine(selected_fast_model, selected_reasoning_model)

if len(engine_bundle) == 6:
  (
      db,
      llm,
      agent_executor,
      query_agent,
      rewrite_agent,
      synthesis_engine,
  ) = engine_bundle
else:
  # Önbellek eski 5'liyi döndürürse sistemi çökertmeden sentez motorunu tamamla
  db, llm, agent_executor, query_agent, rewrite_agent = engine_bundle[:5]
  try:
    from agent import SynthesisEngine

    synthesis_engine = SynthesisEngine(llm)
  except Exception:
    from agent import synthesis_engine

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    **Aktif Motor:** `{profile_badge}`
    - ⚡ *SQL / JSON:* `{selected_fast_model}`
    - 🧠 *Strateji / Sentez:* `{selected_reasoning_model}`
    
    ---
    **Sistem Yetenekleri:**
    - 🎯 *Kök Sebep Analizi (Topics & Products)*
    - 👥 *İlişkisel Demografi (SQL JOIN)*
    - 📉 *Huni Daralması (Funnel Narrowing)*
    - 🛡️ *Duygu Saplantısı Filtresi*
    """
)


# --- 3. MAKRO TEMA VE GRAFİK MOTORU ---
THEME_KEYWORDS = {
    "Satın Alma Sonrası & Teslimat/İade": ["satın alma", "iade", "teslimat", "kargo", "gecikme", "sipariş"],
    "Servis, Garanti & Onarım": ["servis", "garanti", "arıza", "tamir", "parça", "yedek", "onarım", "bakım"],
    "Ürün Kalitesi & Dayanıklılık": ["kalite", "performans", "dayanıklılık", "bozul", "fırın", "süpürge", "ocak", "alet", "küçük ev"],
    "Fiyat, Değer & Kampanya": ["fiyat", "kampanya", "indirim", "pahalı", "ücret", "değer"],
    "Müşteri Hizmetleri & Çağrı": ["müşteri hizmet", "çağrı", "destek", "iletişim", "ulaşım", "telefon"],
    "İnovasyon, Teknoloji & İmaj": ["inovasyon", "teknoloji", "liderlik", "güven", "tasarım", "robot"]
}

def map_micro_to_macro_theme(text: str) -> str:
    t_low = str(text).lower()
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in t_low for kw in kws):
            return theme
    return "Diğer Müşteri Geri Bildirimleri"


def render_generative_ui(result_data, query_json: dict, title_context: str = ""):
    if not result_data:
        st.info("Bu adım için görselleştirilecek veri dönmedi.")
        return

    df = None
    try:
        if isinstance(result_data, str) and result_data.startswith("[("):
            import ast
            raw_tuples = ast.literal_eval(result_data)
            cols = []
            if query_json.get("group_by"):
                cols.extend(query_json["group_by"])
            if query_json.get("aggregates"):
                cols.extend([agg.get("as", "Adet") for agg in query_json["aggregates"]])
            if not cols and raw_tuples:
                cols = [f"Alan_{i+1}" for i in range(len(raw_tuples[0]))]
            df = pd.DataFrame(raw_tuples, columns=cols[:len(raw_tuples[0])] if raw_tuples else None)
        elif isinstance(result_data, list) and len(result_data) > 0 and isinstance(result_data[0], dict):
            df = pd.DataFrame(result_data)
        elif isinstance(result_data, pd.DataFrame):
            df = result_data.copy()
    except Exception:
        df = None

    if df is None or df.empty:
        st.write("**Ham Veri:**", result_data)
        return

    df.columns = [str(c).replace('"', '').replace("'", "").split(".")[-1] for c in df.columns]

    for c in df.columns:
        converted = pd.to_numeric(df[c], errors='coerce')
        if not converted.isna().all() and converted.notna().sum() > len(df) * 0.5:
            df[c] = converted

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    if not numeric_cols and categorical_cols:
        cat_c = categorical_cols[0]
        freq = df[cat_c].value_counts().reset_index()
        freq.columns = [cat_c, "Bahsedilme Sayısı"]
        df = freq
        numeric_cols = ["Bahsedilme Sayısı"]
        categorical_cols = [cat_c]

    val_col = numeric_cols[0] if numeric_cols else df.columns[-1]

    # Senaryo 1: Demografi (Yaş & Cinsiyet)
    has_age = any("age" in c.lower() for c in df.columns)
    has_gen = any("gender" in c.lower() for c in df.columns)
    if has_age and has_gen:
        age_col = [c for c in df.columns if "age" in c.lower()][0]
        gen_col = [c for c in df.columns if "gender" in c.lower()][0]
        fig = px.bar(
            df,
            x=age_col,
            y=val_col,
            color=gen_col,
            barmode="group",
            title=f"{title_context} (Yaş ve Cinsiyet Dağılımı)",
            text=val_col,
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=40), yaxis_title="Bahsedilme Hacmi")
        st.plotly_chart(fig, use_container_width=True)

    # Senaryo 2: Kök Neden ve Konu Dağılımı
    elif categorical_cols:
        cat_col = categorical_cols[0]
        if df[val_col].max() <= 3 and len(df) >= 3:
            df_plot = df.copy()
            df_plot["Makro Tema"] = df_plot[cat_col].apply(map_micro_to_macro_theme)
            df_plot = df_plot.groupby("Makro Tema")[val_col].sum().reset_index()
            df_plot = df_plot.sort_values(by=val_col, ascending=True)
            plot_cat = "Makro Tema"
            st.caption("ℹ️ *Tekil mikro başlıklar karar verilebilir netlik için makro iş temalarına konsolide edilmiştir.*")
        else:
            df[cat_col] = df[cat_col].astype(str).apply(lambda x: (x[:40] + "...") if len(x) > 40 else x)
            df_plot = df.sort_values(by=val_col, ascending=True).tail(8)
            plot_cat = cat_col

        fig = px.bar(
            df_plot,
            x=val_col,
            y=plot_cat,
            orientation="h",
            title=f"{title_context} (Hacim Dağılımı)",
            text=val_col,
            color_discrete_sequence=["#2b5c8f"]
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=340,
            xaxis_title="Bahsedilme Sayısı",
            yaxis_title="",
            margin=dict(l=10, r=30, t=40, b=30),
            yaxis=dict(tickfont=dict(size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    with st.expander("📋 Detaylı Veri Tablosunu İncele", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)


# --- 4. ÇALIŞMA MODLARI ---
mod = st.sidebar.radio(
    "Çalışma Modunu Seçin:",
    [
        "🤖 Otonom İçgörü Modu",
        "👤 Manuel Soru Modu",
        "🧪 Hipotez Doğrulama Modu",
        "🔮 Tahminleme (Predictive) Modu"
    ]
)

# MOD 1: OTONOM İÇGÖRÜ MODU
if mod == "🤖 Otonom İçgörü Modu":
    st.subheader("🤖 Otonom Stratejik İçgörü Keşfi")
    st.write("Ajan, veritabanındaki anomalileri ve operasyonel tıkanıklıkları otonom olarak araştırır.")

    if st.button("🚀 Otonom Taramayı Başlat"):
        with st.spinner("Veritabanı şeması taranıyor ve makro iş problemi belirleniyor..."):
            schema = query_agent.schema
            macro_q = rewrite_agent.generate_macro_question("Tablolar: twitter_tweets, demo_brand_users, demo_brand_predictions")
            st.info(f"**Belirlenen Stratejik Araştırma Sorusu:** {macro_q}")
            _, sub_qs = rewrite_agent.decompose_question(macro_q, schema)
            
        evidence_list = []
        all_query_results = []
        for i, sq in enumerate(sub_qs[:2], 1):
            with st.status(f"Adım {i}: {sq}", expanded=False):
                try:
                    q_res = query_agent.execute_nl_query(sq)
                    evidence_list.append(f"Soru: {sq}\nBulgu: {q_res['result']}\nSQL: {q_res['sql']}")
                    all_query_results.append((sq, q_res))
                    st.code(q_res['sql'], language="sql")
                except Exception as e:
                    evidence_list.append(f"Soru: {sq}\nHata: {e}")

        with st.spinner("Yönetici özeti hazırlanıyor..."):
            combined_evidence = "\n\n".join(evidence_list)
            insight = synthesis_engine.synthesize_executive_summary(macro_q, combined_evidence)
            
            st.markdown("### 📌 Yönetici Özeti (Final Insight)")
            st.success(insight)

            if all_query_results:
                tab_titles = [f"📊 Görsel Analiz (Adım {idx+1})" for idx in range(len(all_query_results))]
                tabs = st.tabs(tab_titles)
                for idx, tab in enumerate(tabs):
                    with tab:
                        sq_text, res_obj = all_query_results[idx]
                        st.caption(f"**Soru:** {sq_text}")
                        render_generative_ui(res_obj["result"], res_obj["json_query"], f"Adım {idx+1}")

# ==============================================================================
# MOD 2: ÇOK TURLU SOHBET VE KONU TAKİBİ MODU (ROADMAP 3A - CHAT THREADING)
# ==============================================================================
elif mod == "👤 Manuel Soru Modu":
    st.subheader("👤 Yönetici Soru ve Takip Modu (Chat Threading)")
    st.caption("Önceki sorularınızı ve filtrelerinizi unutmayan çok turlu konuşma motoru.")

    # Oturum durumlarını (Session State) başlat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_clarification" not in st.session_state:
        st.session_state.pending_clarification = None
    if "active_final_query" not in st.session_state:
        st.session_state.active_final_query = None

    # Sol tarafa sohbeti temizleme butonu ekle
    with st.sidebar:
        if st.button("🗑️ Konuyu Sıfırla (Yeni Oturum)"):
            st.session_state.chat_history = []
            st.session_state.pending_clarification = None
            st.session_state.active_final_query = None
            st.rerun()

    # 1. Önceki konuşma geçmişini ekrana bas
    for idx, turn in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(f"**{turn['user_query']}**")
            if turn.get("resolved_query") and turn["resolved_query"] != turn["user_query"]:
                st.caption(f"*(Bağlamdan türetilen analiz sorusu: {turn['resolved_query']})*")

        with st.chat_message("assistant"):
            st.markdown(turn["insight"])
            if turn.get("all_query_results"):
                tab_titles = [f"📊 Grafik {t_idx+1}" for t_idx in range(len(turn["all_query_results"]))]
                tabs = st.tabs(tab_titles)
                for t_idx, tab in enumerate(tabs):
                    with tab:
                        sq_text, res_obj = turn["all_query_results"][t_idx]
                        st.caption(f"**Soru:** {sq_text}")
                        render_generative_ui(res_obj["result"], res_obj["json_query"], f"T{idx+1}-Adım{t_idx+1}")

    # 2. Netleştirme bekleyen butonlar varsa göster
    if st.session_state.pending_clarification:
        clar_data = st.session_state.pending_clarification
        st.warning(f"🤔 **Hedef Belirleme:** {clar_data['message']}")
        opt_cols = st.columns(len(clar_data["options"]))
        for idx, opt in enumerate(clar_data["options"]):
            with opt_cols[idx]:
                if st.button(opt["label"], key=f"clar_btn_chat_{idx}"):
                    st.session_state.active_final_query = f"{clar_data['base_query']} ({opt['context']})"
                    st.session_state.pending_clarification = None
                    st.rerun()

    # 3. Yeni soru girişi (Chat Input)
    user_input = st.chat_input("Pazarlama stratejisi veya takip sorunuzu yazın (Örn: Peki bunun kadınlar arasındaki oranı ne?)...")

    if user_input:
        schema = query_agent.schema

        # Konuşma hafızasından takip sorusunu çözümle
        with st.spinner("Önceki bağlam taranıyor ve soru netleştiriliyor..."):
            resolved_query = rewrite_agent.contextualize_query(
                user_input, 
                [{"user": t["user_query"], "assistant_summary": t["insight"]} for t in st.session_state.chat_history]
            )

        # Netleştirme gerekiyor mu denetle
        clarification_eval = rewrite_agent.assess_clarification_need(resolved_query, schema)

        if clarification_eval.get("needs_clarification", False) and clarification_eval.get("options"):
            st.session_state.pending_clarification = {
                "base_query": resolved_query,
                "message": clarification_eval.get("clarification_message", "Hangi alana odaklanalım?"),
                "options": clarification_eval.get("options", [])
            }
            st.rerun()
        else:
            st.session_state.pending_clarification = None
            st.session_state.active_final_query = resolved_query
            st.session_state.last_user_raw_input = user_input

    # 4. Soruyu Çalıştır ve Yanıtı Kaydet
    if st.session_state.active_final_query:
        query_to_run = st.session_state.active_final_query
        raw_input = getattr(st.session_state, "last_user_raw_input", query_to_run)
        st.session_state.active_final_query = None

        with st.chat_message("user"):
            st.markdown(f"**{raw_input}**")
            if query_to_run != raw_input:
                st.caption(f"*(Bağlamdan türetilen analiz: {query_to_run})*")

        with st.chat_message("assistant"):
            with st.spinner("Soru alt analiz adımlarına ayrıştırılıyor..."):
                schema = query_agent.schema
                _, sub_qs = rewrite_agent.decompose_question(query_to_run, schema)

            evidence_list = []
            all_query_results = []
            for i, sq in enumerate(sub_qs[:2], 1):
                with st.status(f"Analiz Adımı {i}: {sq}", expanded=False):
                    try:
                        q_res = query_agent.execute_nl_query(sq)
                        evidence_list.append(f"Bulgu: {q_res['result']}")
                        all_query_results.append((sq, q_res))
                        st.code(q_res['sql'], language="sql")
                    except Exception as e:
                        evidence_list.append(f"Hata: {e}")

            with st.spinner("Yönetici özeti sentezleniyor..."):
                combined_evidence = "\n\n".join(evidence_list)
                insight = synthesis_engine.synthesize_executive_summary(query_to_run, combined_evidence)

                st.markdown(insight)

                if all_query_results:
                    tab_titles = [f"📊 Analiz {idx+1}" for idx in range(len(all_query_results))]
                    tabs = st.tabs(tab_titles)
                    for idx, tab in enumerate(tabs):
                        with tab:
                            sq_text, res_obj = all_query_results[idx]
                            st.caption(f"**Araştırılan Soru:** {sq_text}")
                            render_generative_ui(res_obj["result"], res_obj["json_query"], f"Bulgu {idx+1}")

            # Konuşma hafızasına kaydet
            st.session_state.chat_history.append({
                "user_query": raw_input,
                "resolved_query": query_to_run,
                "insight": insight,
                "all_query_results": all_query_results
            })
            st.rerun()


# ==============================================================================
# MOD 3: ÇOKLU VE RAKİP HİPOTEZ DOĞRULAMA MODU (ROADMAP 3E)
# ==============================================================================
elif mod == "🧪 Hipotez Doğrulama Modu":
    st.subheader("🧪 Çoklu ve Rakip Hipotez Doğrulama Motoru")
    st.write("Tekil doğrulama yanlılığını engelleyin: Fikrinizi Sıfır Hipotezi ($H_0$) ve Rakip Hipotezlerle ($H_2$) çapraz sınayın.")

    hyp_input = st.text_input(
        "Sınamak istediğiniz iş hipotezini veya gözlemi girin:",
        value="The sharp decline in Consideration in 2026 means the top of the funnel is narrowing."
    )

    if st.button("⚖️ Hipotezleri Yarıştır ve Test Et") and hyp_input:
        with st.spinner("Yarışan hipotezler ($H_0, H_1, H_2$) türetiliyor ve ayırt edici sorgular planlanıyor..."):
            schema = query_agent.schema
            hyp_dict, test_qs = rewrite_agent.formulate_competing_hypotheses(hyp_input, schema)

        # Hipotezleri Ekranda 3 Kart Olarak Göster
        col_h0, col_h1, col_h2 = st.columns(3)
        with col_h0:
            st.info(f"**Sıfır Hipotezi ($H_0$):**\n\n{hyp_dict.get('H0', 'Tanımlanmadı')}")
        with col_h1:
            st.success(f"**Birincil Hipotez ($H_1$):**\n\n{hyp_dict.get('H1', 'Tanımlanmadı')}")
        with col_h2:
            st.warning(f"**Rakip Hipotez ($H_2$):**\n\n{hyp_dict.get('H2', 'Tanımlanmadı')}")

        st.divider()

        # Ayırt Edici SQL Sorgularını Çalıştır
        evidence_list = []
        all_query_results = []
        for i, tq in enumerate(test_qs[:2], 1):
            with st.status(f"Ayırt Edici Test Aşaması {i}: {tq}", expanded=False):
                try:
                    q_res = query_agent.execute_nl_query(tq)
                    evidence_list.append(f"Test Sorusu: {tq}\nBulgu: {q_res['result']}")
                    all_query_results.append((tq, q_res))
                    st.code(q_res['sql'], language="sql")
                except Exception as e:
                    evidence_list.append(f"Hata: {e}")

        # Sentez ve Karşılaştırmalı Karne
        with st.spinner("Pazarlama alan bilgisi işletiliyor ve Karşılaştırmalı Hipotez Karnesi oluşturuluyor..."):
            combined_evidence = "\n\n".join(evidence_list)
            scorecard_report = synthesis_engine.evaluate_competing_hypotheses(hyp_dict, combined_evidence)

            st.markdown("### 🏆 Hipotez Karşılaştırma Raporu ve Karne")
            st.markdown(scorecard_report)

            if all_query_results:
                tab_titles = [f"📊 Kanıt Verisi (Test {idx+1})" for idx in range(len(all_query_results))]
                tabs = st.tabs(tab_titles)
                for idx, tab in enumerate(tabs):
                    with tab:
                        tq_text, res_obj = all_query_results[idx]
                        st.caption(f"**Ayırt Edici Soru:** {tq_text}")
                        render_generative_ui(res_obj["result"], res_obj["json_query"], f"Kanıt {idx+1}")




# MOD 4: TAHMİNLEME (PREDICTIVE) MODU
elif mod == "🔮 Tahminleme (Predictive) Modu":
    st.subheader("🔮 Gelecek Dönem Projeksiyonu ve Trend Tahmini")
    st.write("Zaman serilerini inceleyerek olası riskleri ve büyüme eğilimlerini öngörün.")

    pred_input = st.text_input(
        "Geleceğini tahmin etmek istediğiniz metrik veya konuyu girin:",
        placeholder="Örn: Önümüzdeki dönemde kargo ve teslimat kaynaklı şikayetler nasıl seyredecek?"
    )

    if st.button("Trend Analizi ve Projeksiyon Üret") and pred_input:
        with st.spinner("Zaman serisi sorguları planlanıyor..."):
            schema = query_agent.schema
            _, trend_qs = rewrite_agent.decompose_predictive_trends(pred_input, schema)

        evidence_list = []
        all_query_results = []
        for i, tq in enumerate(trend_qs[:2], 1):
            with st.status(f"Trend Adımı {i}: {tq}", expanded=False):
                try:
                    q_res = query_agent.execute_nl_query(tq)
                    evidence_list.append(f"Zaman Serisi Bulgusu: {q_res['result']}")
                    all_query_results.append((tq, q_res))
                    st.code(q_res['sql'], language="sql")
                except Exception as e:
                    evidence_list.append(f"Hata: {e}")

        with st.spinner("Tahminleme ve erken uyarı raporu oluşturuluyor..."):
            combined_evidence = "\n\n".join(evidence_list)
            pred_insight = synthesis_engine.synthesize_predictive_insight(pred_input, combined_evidence)

            st.markdown("### 📈 Gelecek Trend Projeksiyonu ve Risk Analizi")
            st.warning(pred_insight)

            if all_query_results:
                tab_titles = [f"📊 Trend Dağılımı (Adım {idx+1})" for idx in range(len(all_query_results))]
                tabs = st.tabs(tab_titles)
                for idx, tab in enumerate(tabs):
                    with tab:
                        tq_text, res_obj = all_query_results[idx]
                        st.caption(f"**Trend Sorusu:** {tq_text}")
                        render_generative_ui(res_obj["result"], res_obj["json_query"], f"Trend {idx+1}")