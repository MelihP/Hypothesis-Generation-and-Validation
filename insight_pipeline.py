import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. API Anahtarı ve Veritabanı Bağlantısı
os.environ["GROQ_API_KEY"] = "" # Lütfen kendi anahtarını yapıştır
db = SQLDatabase.from_uri("sqlite:///insight_generation_bot.db")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Ajanı oluşturuyoruz
agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=False)

print("\n--- INSIGHT GENERATION BOT BAŞLATILIYOR ---\n")

# 2. Aşama 1: HL-G (Yüksek Seviyeli Stratejik Soru)
short_D_info = "Bu veritabanı, bir markanın sosyal medya performansını, tüketici şikayetlerini, duygu analizlerini (emotion_analysis) ve demografik (demographics) yapılarını içermektedir."

hl_prompt = PromptTemplate.from_template(
    "Sen uzman bir pazarlama direktörüsün. Veritabanı özeti: {info}\n"
    "Lütfen marka sağlığını ve müşteri şikayetlerini analiz etmek için tek bir tane vizyoner ve stratejik iş sorusu üret. "
    "Sadece soruyu yaz."
)
hl_chain = hl_prompt | llm
macro_question = hl_chain.invoke({"info": short_D_info}).content
print(f"[1. AŞAMA - HL-G] Üretilen Stratejik Soru:\n{macro_question}\n")


# 3. Aşama 2: LL-G (Alt Sorgulara Bölme - ŞEMA OPTİMİZASYONU EKLENDİ)
# Veritabanının tam şemasını ve örnek satırlarını (D_schema) çekiyoruz
d_schema = db.get_table_info()

ll_prompt = PromptTemplate.from_template(
    "Sen bir veri analistisin. Veritabanının tam şeması ve örnek verileri aşağıdadır:\n"
    "{schema}\n\n"
    "Stratejik soru: {question}\n"
    "Bu stratejik soruyu cevaplamak için YUKARIDAKİ ŞEMADA YER ALAN SÜTUNLARI VE KATEGORİLERİ (örneğin örnek verilerdeki yaş gruplarını veya duyguları) BİREBİR KULLANARAK 2 adet spesifik, net ve kısa alt soruya böl.\n"
    "Kategorileri asla dışarıdan uydurma, sadece şemadaki örnek verilerde gördüğün kesin değerleri (örn: '18-29', '40>') kullan.\n"
    "Sadece soruları madde imi (-) kullanarak alt alta yaz, başka hiçbir açıklama yapma."
)
ll_chain = ll_prompt | llm
sub_questions_text = ll_chain.invoke({"question": macro_question, "schema": d_schema}).content
print(f"[2. AŞAMA - LL-G] Şema Destekli SQL Alt Soruları:\n{sub_questions_text}\n")


# 4. Aşama 3: Query Agent (Veritabanından Gerçekleri Çekme)
print("[3. AŞAMA - DOĞRULAMA] Ajan veritabanını tarıyor (Bu işlem 10-15 saniye sürebilir)...\n")
facts = []
for q in sub_questions_text.split('\n'):
    if q.strip().startswith('-'):
        soru = q.replace('-', '').strip()
        print(f"Sorgulanıyor: {soru}")
        try:
            ans = agent_executor.invoke({"input": soru})["output"]
            facts.append(ans)
            print(f"Bulunan Veri: {ans}\n")
        except Exception as e:
            print("Veri bulunamadı.\n")


# 5. Aşama 4: Özetleme ve Halüsinasyon Filtresi
facts_str = "\n".join(facts)
summary_prompt = PromptTemplate.from_template(
    "Aşağıdaki 'Doğrulanmış Veri Gerçeklerini' kullanarak, yöneticiler için maksimum 3 cümlelik, "
    "stratejik ve eyleme dönüştürülebilir bir pazarlama içgörüsü (insight) yaz.\n"
    "KURALLAR:\n"
    "1. Sadece verilen rakamları kullan, asla dışarıdan bir bilgi veya oran uydurma (halüsinasyon yapma).\n"
    "2. Maddeleme kullanma, paragraf şeklinde stratejik bir dil kullan.\n\n"
    "Veri Gerçekleri:\n{facts}"
)
summary_chain = summary_prompt | llm
final_insight = summary_chain.invoke({"facts": facts_str}).content

print(f"==========================================")
print(f"🎯 [4. AŞAMA - FİNAL PAZARLAMA İÇGÖRÜSÜ] 🎯")
print(f"==========================================")
print(final_insight)