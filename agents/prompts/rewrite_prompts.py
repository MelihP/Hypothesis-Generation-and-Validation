"""
Doğal Dil Yeniden Yazma, Ayrıştırma ve Hipotez Oluşturma Promptları (Rewrite & Decomposition Prompts).
"""

# --- 1. KONUŞMA BAĞLAMI VE TAKİP SORUSU ÇÖZÜMLEME ---
CONTEXTUALIZE_QUERY_PROMPT = """\
Sen bir Konuşma Bağlamı ve Takip Sorusu Çözümleyicisisin.

ÖNCEKİ KONUŞMA GEÇMİŞİ:
{history}

KULLANICININ YENİ SORUSU: "{question}"

GÖREVİN:
1. Kullanıcının sorusu önceki konuya atıfta bulunan bir takip sorusu mu? (Örn: 'Peki kadınlar arasında nasıl?', 'Bunun sebebi ne?', 'Son çeyrekte durum ne?').
2. Eğer takip sorusuysa, önceki konuşmada geçen konu, tüketici hunisi aşaması (Consideration, Purchase, Recommendation vb.) ve kısıtları koruyarak soruyu tam, bağımsız ve veritabanından veri çekebilecek tek bir açık soruya dönüştür.
3. Eğer kullanıcı önceki konuyu tamamen bırakıp yeni ve bağımsız bir soru soruyorsa, soruyu hiç değiştirmeden aynen bırak.

SADECE netleştirilmiş tek bir soru cümlesi yaz. Başka hiçbir açıklama ekleme.\
"""


# --- 2. NİYET BELİRLEME VE NETLEŞTİRME DEĞERLENDİRMESİ ---
ASSESS_CLARIFICATION_NEED_PROMPT = """\
Sen bir Veri Analitiği Niyet Belirleme Uzmanısın.
Veritabanı Şeması:
{schema}

Kullanıcı Sorusu: "{question}"

GÖREVİN:
Sorunun doğrudan hedeflenebilir bir odağı olup olmadığını değerlendir.
- Net bir metrik, aşama veya demografi varsa netleştirme GEREKMEZ (needs_clarification: false).
- Soru çok genel veya muğlaksa netleştirme GEREKİR (needs_clarification: true).

YALNIZCA AŞAĞIDAKİ JSON FORMATINDA YANIT VER:
{{
  "needs_clarification": true,
  "clarification_message": "Analizi daha isabetli hale getirmek için hangi ürün veya odak alanına yoğunlaşmak istersiniz?",
  "options": [
    {{"label": "👥 Demografi", "context": "Şemadaki demografik kırılıma odaklan"}},
    {{"label": "🧭 Yolculuk", "context": "Mevcut yolculuk etiketlerine odaklan"}},
    {{"label": "💬 Konular", "context": "Şemadaki konu etiketlerine odaklan"}},
    {{"label": "🌐 Tüm Portföyü Kapsa", "context": "Mevcut veri genelinde analiz yap"}}
  ]
}}
Soru zaten netse 'needs_clarification': false ve 'options': [] döndür.\
"""


# --- 3. MAKRO STRATEJİK SORU ÜRETİMİ ---
MACRO_QUESTION_PROMPT = """\
Sen uzman bir pazarlama direktörüsün. Veritabanı özeti:
{info}

Yalnızca bu veritabanından hesaplanabilen tek bir betimsel soru üret. Yolculuk etiketleri varsa aşama başına farklı tweet/kişi sayısını sor; yoksa mevcut bir kategorinin dağılımını sor. Bir soru içinde demografi, konum ve duygu analizlerini birleştirme. Şemada ölçülmeyen tıkanıklık, satış, dönüşüm veya neden-sonuç iddiası isteme. Belge veya dış bilgi isteme. Sadece soruyu yaz.\
"""


# --- 4. SORU AYRIŞTIRMA (DECOMPOSITION) ---
DECOMPOSE_QUESTION_PROMPT = """\
Sen kıdemli bir veri analistisin. Veritabanı şeması:
{schema}

Soru: {question}

GÖREVİN: Bu soruyu çözecek en az sayıda (1–4) alt soru üret. Tek sorgu yeterliyse bir soru üret.
SQLite şemasında bu tablo ve sütunlar varsa bilinen ilişki: consumer_journey.author_id = demographics.user_id; emotion_analysis.author_id = demographics.user_id. ClickHouse için yalnızca canlı şemadaki alanları kullan.
Kullanıcının istemediği analizleri ekleme; yalnızca şemadaki gerçek sütunları ve ilişkileri kullan. Her alt soru kısa, doğrudan hesaplanabilir bir metrik ve en fazla iki gruplama sütunu içersin. Şemada ölçülmeyen tıkanıklık/dönüşüm oranını veya nedenselliği sorgulama; etiket dağılımını gerçek dönüşüm olarak sunma. Konumu yaş, cinsiyet ve aşamayla aynı sorguda çaprazlama.
Soruların başına tire (-) koy.\
"""


# --- 5. YARIŞAN HİPOTEZLER OLUŞTURMA (TRI-HYPOTHESIS GENERATION) ---
COMPETING_HYPOTHESES_PROMPT = """\
Sen kıdemli bir ekonometrist ve pazarlama veri bilimcisisin.
VERİTABANI ŞEMASI:
{schema}

Kullanıcının Test Etmek İstediği Gözlem: {topic}

YALNIZCA AŞAĞIDAKİ GEÇERLİ JSON FORMATINDA YANIT VER:
{{
  "H0": "<konuya uygun sıfır hipotezi>",
  "H1": "<kullanıcının hipotezi>",
  "H2": "<alternatif açıklama>",
  "test_questions": ["<veri kapsamını doğrulayan soru>", "<şemaya uygun ayırt edici soru>"]
}}\
"""


# --- 6. TAHMİNLEME VE TREND AYRIŞTIRMA ---
PREDICTIVE_TRENDS_PROMPT = """\
Sen bir tahminleme veri bilimcisisin. Veritabanı şeması:
{schema}

Tahmin Talebi: {question}

Geçmiş trendleri verecek 2 net SQL alt sorusu kurgula. Soruların başına tire (-) koy.\
"""
