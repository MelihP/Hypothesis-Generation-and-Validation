import streamlit as st
import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- 1. AYARLAR VE VERİTABANI BAĞLANTISI ---
os.environ["GROQ_API_KEY"] = ""

@st.cache_resource
def get_db_and_agent():
    db = SQLDatabase.from_uri("sqlite:///insight_generation_bot.db")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # zero-shot yerine daha stabil olan tool-calling mimarisine geçiyoruz
    agent = create_sql_agent(
        llm, 
        db=db, 
        agent_type="openai-tools", 
        verbose=False
    )
    return db, llm, agent

db, llm, agent_executor = get_db_and_agent()

# --- 2. WEB ARAYÜZÜ TASARIMI ---
st.set_page_config(page_title="Pazarlama İçgörü Motoru", page_icon="📊", layout="centered")

st.title("📊 Pazarlama İçgörü Motoru")
st.markdown("Veritabanınızı otonom tarayabilir veya doğrudan kendi stratejik sorunuzu sorabilirsiniz.")
st.divider()

# --- YENİ EKLENEN ÖZELLİK: MOD SEÇİMİ ---
calisma_modu = st.radio(
    "Nasıl bir içgörü istiyorsunuz?",
    ["🤖 Otonom Mod (Yapay Zeka Sorsun ve Çözsün)", "👤 Manuel Mod (Kendi Sorunuzu Sorun)"],
    horizontal=True
)

manuel_soru = ""
if calisma_modu == "👤 Manuel Mod (Kendi Sorunuzu Sorun)":
    manuel_soru = st.text_input("Pazarlama / Strateji sorunuzu buraya yazın:", placeholder="Örn: Gençlerde (18-29) en çok hangi duygu hakim?")

st.write("") # Boşluk

if st.button("🚀 İçgörü Üret", use_container_width=True):
    
    # Manuel Mod için boşluk kontrolü
    if calisma_modu == "👤 Manuel Mod (Kendi Sorunuzu Sorun)" and not manuel_soru:
        st.warning("Lütfen üret butonuna basmadan önce bir soru girin!")
        st.stop()
    
    # 1. Aşama (HL-G veya Manuel Soru Ataması)
    with st.status("🧠 1. Aşama: Stratejik soru belirleniyor...", expanded=True) as status:
        if calisma_modu == "🤖 Otonom Mod (Yapay Zeka Sorsun ve Çözsün)":
            short_D_info = "Bir markanın sosyal medya performansı, müşteri şikayetleri ve demografik bilgileri."
            hl_prompt = PromptTemplate.from_template(
                "Sen bir pazarlama direktörüsün. Veritabanı özeti: {info}\n"
                "Sadece marka sağlığını ölçecek vizyoner tek bir iş sorusu üret."
            )
            macro_question = (hl_prompt | llm).invoke({"info": short_D_info}).content
        else:
            macro_question = manuel_soru
            
        st.write(f"**Odaklanılan Soru:** {macro_question}")
        status.update(label="✅ 1. Aşama Tamamlandı!", state="complete", expanded=False)

    # 2. Aşama (LL-G - İNSAN DİLİNİ ANLAYAN GELİŞMİŞ PROMPT)
    with st.status("⚙️ 2. Aşama: Soru veritabanı diline çevriliyor...", expanded=True) as status:
        d_schema = db.get_table_info()
        ll_prompt = PromptTemplate.from_template(
            "Sen kıdemli bir veri analistisin. Veritabanının şeması:\n{schema}\n\n"
            "Kullanıcının sorduğu stratejik soru: {question}\n\n"
            "GÖREVİN: Bu insan sorusunu SQL'in anlayabileceği 2 net alt soruya bölmek.\n"
            "PROMPT MÜHENDİSLİĞİ KURALLARI:\n"
            "1. Kullanıcı şemadaki isimleri bilmez. Eğer kullanıcı 'gençler' diyorsa, sen bunu şemadaki '18-29' age_group olarak çevir.\n"
            "2. Sorular kesinlikle somut (COUNT, miktar, hangi duygu) hedefler içersin.\n"
            "3. Soruların sonuna (?) koy.\n"
            "4. Sadece soruları alt alta yaz (- ile başla)."
        )
        sub_questions_text = (ll_prompt | llm).invoke({"question": macro_question, "schema": d_schema}).content
        st.markdown(sub_questions_text)
        status.update(label="✅ 2. Aşama Tamamlandı!", state="complete", expanded=False)

    # 3. Aşama (Query Agent)
    with st.status("🔍 3. Aşama: Ajan veritabanını tarayıp kanıtları topluyor...", expanded=True) as status:
        facts = []
        for line in sub_questions_text.split('\n'):
            clean_line = line.strip()
            if not clean_line: continue
            
            if "?" in clean_line:
                soru = clean_line.lstrip("-*0123456789. ").strip()
                if soru:
                    st.write(f"👉 *Sorgulanıyor:* {soru}")
                    try:
                        ans = agent_executor.invoke({"input": soru})["output"]
                        facts.append(ans)
                        st.success(f"**Bulunan Kanıt:** {ans}")
                    except Exception as e:
                        st.error(f"Veri çekilemedi: {e}")
                        
        status.update(label="✅ 3. Aşama Tamamlandı!", state="complete", expanded=False)

    # 4. Aşama (Summarization)
    if facts:
        with st.spinner("📝 4. Aşama: Final Yönetici Raporu Hazırlanıyor..."):
            facts_str = "\n".join(facts)
            summary_prompt = PromptTemplate.from_template(
                "Aşağıdaki 'Doğrulanmış Veri Gerçekleri'ni kullanarak kullanıcının sorduğu ana soruya ({main_q}) cevap veren "
                "maksimum 3 cümlelik pazarlama içgörüsü yaz.\n"
                "KURALLAR: 1. Asla uydurma (halüsinasyon) yapma. 2. Paragraf şeklinde stratejik bir dil kullan.\n\nVeri Gerçekleri:\n{facts}"
            )
            final_insight = (summary_prompt | llm).invoke({"facts": facts_str, "main_q": macro_question}).content

        st.divider()
        st.subheader("🎯 Yönetici Özeti (Final Insight)")
        st.info(final_insight, icon="💡")
        st.balloons()