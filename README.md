# 📊 AI-Powered Hybrid Marketing Insight Engine

Bu proje, yapılandırılmış (SQL) ve yapılandırılmamış (RAG/Vektör) verileri aynı anda analiz edebilen, LangChain ve GPT-4o tabanlı gelişmiş bir kurumsal veri analizi platformudur. Pazarlama yöneticileri için otonom içgörüler üretir, veri odaklı hipotezleri test eder ve geleceğe yönelik projeksiyonlar sunar.

## 🚀 Özellikler

Sistem 4 farklı otonom modda çalışmaktadır:
- **🤖 Otonom İçgörü Modu:** Veritabanını kendi kendine tarayarak en kritik vizyoner pazarlama sorusunu bulur ve yanıtlar.
- **👤 Manuel Soru Modu:** Kullanıcının girdiği stratejik soruları, SQL ve Doküman (RAG) araçlarını yönlendirerek hibrit olarak çözer.
- **🧪 Hipotez Doğrulama Modu:** Ortaya atılan bir fikri (Örn: "Gençler kargo sürecinden şikayetçi") verilerle test eder ve bilimsel olarak "Doğrulandı / Çürütüldü" şeklinde raporlar.
- **🔮 Tahminleme (Predictive AI) Modu:** Geçmiş verilere ve trendlere bakarak gelecek ay/dönem için LLM tabanlı istatistiksel projeksiyonlar yapar.
- **📈 Generative UI:** Çıkan analiz sonuçlarını otomatik olarak algılar ve Plotly kullanarak interaktif pasta/bar grafiklerine dönüştürür.

## 🛠️ Kullanılan Teknolojiler
- **Python & Streamlit:** Web arayüzü ve kullanıcı deneyimi.
- **LangChain:** LLM orkestrasyonu, Tool çağrıları ve Çoklu Ajan (Agentic) yapı.
- **OpenAI (GPT-4o):** İleri düzey akıl yürütme, SQL query yazma ve metin özetleme.
- **Pinecone (RAG):** Yapılandırılmamış şirket dokümanları için vektör veritabanı araması.
- **SQLite & Pandas:** Yapılandırılmış veri yönetimi ve analizi.
- **Plotly:** Dinamik veri görselleştirme.

## ⚙️ Kurulum ve Çalıştırma

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone https://github.com/MelihP/Hypothesis-Generation-and-Validation/tree/main