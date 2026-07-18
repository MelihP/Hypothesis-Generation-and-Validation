import streamlit as st
from langchain_core.prompts import PromptTemplate
# Modüler mimari: Hibrit Ajanı (SQL + RAG) agent.py dosyasından çağırıyoruz!
from agent import db, llm, agent_executor
import json
import plotly.express as px
import pandas as pd

# --- WEB ARAYÜZÜ TASARIMI ---
st.set_page_config(page_title="Hibrit Pazarlama İçgörü Motoru", page_icon="📊", layout="centered")

st.title("📊 Gelişmiş Hibrit Analiz Platformu")
st.markdown("""
Bu sistem, GPT-4o zekasıyla hem **SQL veritabanınızı** (sayılar ve demografi) hem de **Kurumsal Dokümanlarınızı** (RAG/Pinecone) aynı anda tarayabilir. Otomatize hipotez doğrulama çerçevesi işleterek fikirlerinizi bilimsel olarak sınayabilirsiniz.
""")
st.divider()

## --- 4 SEÇENEKLİ ÇALIŞMA MODU ---
calisma_modu = st.radio(
    "Çalışma Modunu Seçin:",
    [
        "🤖 Otonom İçgörü Modu", 
        "👤 Manuel Soru Modu", 
        "🧪 Hipotez Doğrulama Modu",
        "🔮 Tahminleme (Predictive) Modu" # YENİ EKLENEN MOD
    ],
    horizontal=True
)

# Seçilen moda göre dinamik girdi alanları
manuel_soru = ""
if calisma_modu == "👤 Manuel Soru Modu":
    manuel_soru = st.text_input("Pazarlama / Strateji sorunuzu buraya yazın:", placeholder="Örn: 18-29 yaş grubunun en çok şikayet ettiği ürün hangisi...")
elif calisma_modu == "🧪 Hipotez Doğrulama Modu":
    manuel_soru = st.text_input("Sınamak / doğrulamak istediğiniz ana fikri veya konuyu yazın:", placeholder="Örn: Genç kullanıcılar kargo süreçlerinden çok, dijital deneyimlerde sorun yaşıyor.")
elif calisma_modu == "🔮 Tahminleme (Predictive) Modu":
    manuel_soru = st.text_input("Gelecek projeksiyonunu görmek istediğiniz konuyu yazın:", placeholder="Örn: Önümüzdeki 3 ay içinde 'öfke' duygusundaki trend ne olacak?")

st.write("") 

if st.button("🚀 Analizi Başlat", use_container_width=True):
    
    if calisma_modu in ["👤 Manuel Soru Modu", "🧪 Hipotez Doğrulama Modu"] and not manuel_soru:
        st.warning("Lütfen analizi başlatmadan önce ilgili alanı doldurun!")
        st.stop()
    
    # -------------------------------------------------------------------------
    # --- AKIŞ A: MOD 1 (OTONOM) & MOD 2 (MANUEL SORU) ---
    # -------------------------------------------------------------------------
    if calisma_modu in ["🤖 Otonom İçgörü Modu", "👤 Manuel Soru Modu"]:
        
        with st.status("🧠 1. Aşama: Stratejik makro soru yapılandırılıyor...", expanded=True) as status:
            if calisma_modu == "🤖 Otonom İçgörü Modu":
                short_D_info = "Bir markanın sosyal medya performansı, müşteri şikayetleri ve demografik bilgileri."
                hl_prompt = PromptTemplate.from_template(
                    "Sen uzman bir pazarlama direktörüsün. Veritabanı özeti: {info}\n"
                    "Lütfen marka sağlığını analiz etmek için vizyoner tek bir iş sorusu üret. Sadece soruyu yaz."
                )
                macro_question = (hl_prompt | llm).invoke({"info": short_D_info}).content
            else:
                macro_question = manuel_soru
                
            st.write(f"**Odaklanılan Soru:** {macro_question}")
            status.update(label="✅ 1. Aşama: Soru Belirlendi!", state="complete", expanded=False)

        with st.status("⚙️ 2. Aşama: Analiz rotası çiziliyor (SQL & RAG Yönlendirmesi)...", expanded=True) as status:
            d_schema = db.get_table_info()
            ll_prompt = PromptTemplate.from_template(
                "Sen kıdemli bir veri analistisin. Veritabanının şeması:\n{schema}\n\n"
                "Stratejik soru: {question}\n\n"
                "GÖREVİN: Bu soruyu çözmek için ajana rehberlik edecek 2 net alt soru kurgula.\n"
                "BİLGİ YÖNLENDİRMESİ:\n"
                "1. Eğer soru sayılar, oranlar, demografi veya duygularla ilgiliyse bunu ŞEMADAKİ sütunlara göre SQL sorusuna çevir.\n"
                "2. Eğer soru şirket politikaları, vizyon metinleri veya uzun dokümanlarla ilgiliyse bunu 'dokuman_arama_araci' ile çözülecek bir soruya çevir.\n"
                "3. KRİTİK: Soruların başına mutlaka tire (-) işareti koyarak liste halinde yaz."
            )
            sub_questions_text = (ll_prompt | llm).invoke({"question": macro_question, "schema": d_schema}).content
            st.markdown(sub_questions_text)
            status.update(label="✅ 2. Aşama: Alt Sorular ve Rota Hazır!", state="complete", expanded=False)

        with st.status("🔍 3. Aşama: GPT-4o Hibrit Ajanı çalışıyor (Veri ve Doküman Taraması)...", expanded=True) as status:
            facts = []
            # Sinyal kaybını önlemek için güvenli ayrıştırıcı (tire ile başlayanları alır)
            for line in sub_questions_text.split('\n'):
                if line.strip().startswith('-'):
                    soru = line.lstrip("-* ").strip()
                    if soru:
                        st.write(f"👉 *Araştırılıyor:* {soru}")
                        try:
                            ans = agent_executor.invoke({"input": soru})["output"]
                            facts.append(ans)
                            st.success(f"**Bulunan Kanıt:** {ans}")
                        except Exception as e:
                            st.error(f"Veri çekilemedi: {e}")
            
            if not facts: st.warning("⚠️ Ne veritabanından ne de dokümanlardan kanıt toplanamadı.")
            status.update(label="✅ 3. Aşama: Kanıt Toplama Tamamlandı!", state="complete", expanded=False)

        if facts:
            with st.spinner("📝 4. Aşama: Yönetici Raporu Hazırlanıyor..."):
                facts_str = "\n".join(facts)
                summary_prompt = PromptTemplate.from_template(
                    "Aşağıdaki 'Doğrulanmış Veri Gerçeklerini' kullanarak, yöneticiler için 3 cümlelik, "
                    "stratejik bir pazarlama içgörüsü yaz.\n\n"
                    "KRİTİK BİRİM VE MATEMATİK KURALI:\n"
                    "1. Gelen sayılar (Örn: 3390, 1879) kişi/hacim adetleridir. Oran değildir. Başına % işareti KOYMA.\n"
                    "2. Gelen ondalık sayılar (Örn: 0.70) aslında %70 demektir. Yöneticiler için % formatına çevir.\n\n"
                    "Veri Gerçekleri:\n{facts}"
                )
                final_insight = (summary_prompt | llm).invoke({"facts": facts_str}).content

            st.divider()
            st.subheader("🎯 Yönetici Özeti (Final Insight)")
            st.info(final_insight, icon="💡")
            st.balloons()

            # --- YENİ EKLENEN GRAFİK ÇİZME (GENERATIVE UI) BÖLÜMÜ ---
            with st.spinner("📊 5. Aşama: Dinamik Grafik Çiziliyor..."):
                chart_prompt = PromptTemplate.from_template(
                    "Sen bir veri görselleştirme uzmanısın. Aşağıdaki veri gerçeklerine bakarak bir grafik çizmek için JSON formatında veri üret.\n"
                    "Eğer veriler oran veya dağılım içeriyorsa (örn: yaş grupları, cinsiyet) 'pie' (pasta) grafiği seç.\n"
                    "Eğer veriler miktar veya hacim kıyaslamasıysa 'bar' (çubuk) grafiği seç.\n\n"
                    "KURALLAR:\n"
                    "1. Sadece geçerli bir JSON formatı döndür. Başında veya sonunda (```json) gibi markdown işaretleri OLMASIN.\n"
                    "2. Asla açıklama metni yazma.\n\n"
                    "Veriler:\n{facts}\n\n"
                    "Beklenen Çıktı Formatı:\n"
                    "{{\n"
                    "  \"title\": \"Grafik Başlığı\",\n"
                    "  \"type\": \"bar\", \n"
                    "  \"labels\": [\"Kategori 1\", \"Kategori 2\"],\n"
                    "  \"values\": [10, 20]\n"
                    "}}"
                )
                chart_json_str = (chart_prompt | llm).invoke({"facts": facts_str}).content
                
            try:
                # JSON metnindeki olası markdown kalıntılarını temizle
                clean_json = chart_json_str.replace("```json", "").replace("```", "").strip()
                chart_data = json.loads(clean_json)

                # JSON verisini Pandas tablosuna çevir
                df_chart = pd.DataFrame({
                    "Kategori": chart_data["labels"],
                    "Değer": chart_data["values"]
                })

                # Ajanın seçtiği grafik türüne (pie veya bar) göre çizim yap
                if chart_data["type"] == "pie":
                    fig = px.pie(df_chart, names="Kategori", values="Değer", title=chart_data["title"])
                else:
                    fig = px.bar(df_chart, x="Kategori", y="Değer", title=chart_data["title"])
                
                # Grafiği ekrana bas
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.info("Bu veri seti görselleştirme için yeterli sayısal kategori içermiyor.")

    # -------------------------------------------------------------------------
    # --- AKIŞ B: MOD 3 (GELİŞMİŞ HİPOTEZ DOĞRULAMA MOTORU) ---
    # -------------------------------------------------------------------------
    elif calisma_modu == "🧪 Hipotez Doğrulama Modu":
        
        with st.status("🧠 1. Aşama: Otomatik Hipotez Yapılandırılıyor...", expanded=True) as status:
            d_schema = db.get_table_info()
            hyp_prompt = PromptTemplate.from_template(
                "Sen kıdemli bir veri bilimcisisin. Veritabanı şeması:\n{schema}\n\n"
                "Araştırma Konusu: {question}\n\n"
                "GÖREVİN: Bu konuyu test etmek için şemadaki sütunları baz alan tek bir Alternatif Hipotez (H1) üretmek "
                "ve SQL ajanının test edeceği 2 somut alt soru kurgulamak.\n\n"
                "ÇOK ÖNEMLİ KURALLAR:\n"
                "1. Aradığın veriler farklı tablolardaysa 'Tabloları JOIN yaparak birleştirin' şeklinde açık talimat ekle.\n"
                "2. Sorular kesinlikle matematiksel (COUNT, MAX, AVG) olsun. Ham tweet metni çekme (LIMIT hatası almamak için).\n"
                "3. Hipotezini 'EN ÇOK' gibi kesinleyici kelimeler yerine, daha esnek istatistiksel kavramlar üzerine kur.\n\n"
                "FORMAT KURALI:\n"
                "Hipotez (H1): [Hipotez cümlesi]\n"
                "- [1. net SQL sorusu]\n"
                "- [2. net SQL sorusu]"
            )
            hyp_text = (hyp_prompt | llm).invoke({"question": manuel_soru, "schema": d_schema}).content
            st.markdown(hyp_text)
            
            # Sinyal kaybını önlemek için güvenli ayrıştırıcı (tire ile başlayanları alır)
            sub_questions = [line.lstrip("-* ").strip() for line in hyp_text.split('\n') if line.strip().startswith('-')]
            status.update(label="✅ 1. Aşama: Hipotez Kurgulandı!", state="complete", expanded=False)

        with st.status("🔍 2. Aşama: Hipotez GPT-4o ile test ediliyor...", expanded=True) as status:
            facts = []
            for soru in sub_questions:
                if soru:
                    st.write(f"👉 *Test Ediliyor:* {soru}")
                    try:
                        ans = agent_executor.invoke({"input": soru})["output"]
                        facts.append(ans)
                        st.success(f"**Bulunan Kanıt:** {ans}")
                    except Exception as e:
                        st.error(f"Veri çekilemedi: {e}")
                        
            status.update(label="✅ 2. Aşama: Kanıtlar Toplandı!", state="complete", expanded=False)

        if facts:
            with st.spinner("⚖️ 3. Aşama: Bilimsel Doğrulama Kararı Veriliyor..."):
                facts_str = "\n".join(facts)
                val_prompt = PromptTemplate.from_template(
                    "Aşağıda ortaya atılan hipotez ve SQL ajanının getirdiği gerçekler yer alıyor.\n\n"
                    "Hipotez:\n{hypothesis}\n\n"
                    "Toplanan Veri Gerçekleri:\n{facts}\n\n"
                    "GÖREVİN: Verileri inceleyerek hipoteze karar vermek. Seçeneklerin:\n"
                    "1. 'Doğrulandı'\n2. 'Çürütüldü'\n3. 'Kısmen Doğrulandı (Partially Validated)'\n\n"
                    "Yöneticiler için maksimum 3 cümlelik rapor yaz. İlk cümlen hipotezin akıbetini açıkça belirtsin."
                )
                validation_report = (val_prompt | llm).invoke({"hypothesis": hyp_text, "facts": facts_str}).content

            st.divider()
            st.subheader("🎯 Hipotez Doğrulama Sonucu")
            st.info(validation_report, icon="⚖️")
            st.balloons()


    # -------------------------------------------------------------------------
    # --- AKIŞ C: MOD 4 (TAHMİNLEME VE PROJEKSİYON MOTORU - PREDICTIVE AI) ---
    # -------------------------------------------------------------------------
    elif calisma_modu == "🔮 Tahminleme (Predictive) Modu":
        
        with st.status("🧠 1. Aşama: Zaman Serisi ve Trend Analizi Kurgulanıyor...", expanded=True) as status:
            d_schema = db.get_table_info()
            pred_prompt = PromptTemplate.from_template(
                "Sen bir tahminleme (predictive) veri bilimcisisin. Veritabanı şeması:\n{schema}\n\n"
                "Kullanıcının Tahmin Talebi: {question}\n\n"
                "GÖREVİN: Geleceği tahmin edebilmemiz için bize GEÇMİŞ TRENDLERİ verecek 2 net SQL alt sorusu kurgulamak.\n"
                "KURALLAR:\n"
                "1. Zaman (date, timestamp, month vb.) sütunları varsa mutlaka onlara göre grupla (GROUP BY).\n"
                "2. Eğer zaman sütunu yoksa, veriyi büyüklük veya kategori bazında sıralayarak (ORDER BY) bir trend yakalamaya çalış.\n"
                "3. Soruların başına tire (-) koyarak liste halinde ver."
            )
            pred_text = (pred_prompt | llm).invoke({"question": manuel_soru, "schema": d_schema}).content
            st.markdown(pred_text)
            
            sub_questions = [line.lstrip("-* ").strip() for line in pred_text.split('\n') if line.strip().startswith('-')]
            status.update(label="✅ 1. Aşama: Trend Sorguları Hazır!", state="complete", expanded=False)

        with st.status("🔍 2. Aşama: Geçmiş Veriler Toplanıyor...", expanded=True) as status:
            facts = []
            for soru in sub_questions:
                if soru:
                    st.write(f"👉 *Sorgulanıyor:* {soru}")
                    try:
                        ans = agent_executor.invoke({"input": soru})["output"]
                        facts.append(ans)
                        st.success(f"**Bulunan Geçmiş Veri:** {ans}")
                    except Exception as e:
                        st.error(f"Veri çekilemedi: {e}")
                        
            status.update(label="✅ 2. Aşama: Veriler Toplandı!", state="complete", expanded=False)

        if facts:
            with st.spinner("🔮 3. Aşama: LLM Tabanlı Tahminleme (Predictive) Motoru Çalışıyor..."):
                facts_str = "\n".join(facts)
                forecast_prompt = PromptTemplate.from_template(
                    "Sen gelişmiş bir Tahminleme (Predictive) modelisin.\n"
                    "Aşağıdaki geçmiş verilere bakarak matematiksel ve mantıksal bir gelecek projeksiyonu yap.\n\n"
                    "Geçmiş Veriler:\n{facts}\n\n"
                    "GÖREVİN:\n"
                    "1. Gidişatı analiz et.\n"
                    "2. Yakın gelecek (örn: önümüzdeki ay/çeyrek) için tahmini bir metrik veya oransal değişim (örn: %15 artış beklentisi) ver.\n"
                    "3. Bu tahmini kırmak (iyileştirmek) için yöneticilere 1 adet acil eylem planı sun.\n\n"
                    "Format: 3 maddelik kısa ve son derece profesyonel bir rapor."
                )
                forecast_report = (forecast_prompt | llm).invoke({"facts": facts_str}).content

            st.divider()
            st.subheader("🔮 Gelecek Projeksiyonu (Predictive Forecast)")
            st.info(forecast_report, icon="📈")           