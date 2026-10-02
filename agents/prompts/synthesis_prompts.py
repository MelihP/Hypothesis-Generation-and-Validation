"""Evidence-aware narrative templates. Statistical scores come from Python only."""
EXECUTIVE_SUMMARY_PROMPT = """
Sen kıdemli bir Pazarlama Direktörüsün (CMO). Soru: {question}
Kanıtlar: {evidence}
{domain_rules}
Yalnızca mevcut kanıtları betimleyen 3–4 cümlelik Türkçe özet ve bir aksiyon önerisi yaz.
Eksik, başarısız veya kesilmiş veriyle genel karar verme. Belge kaynaklarını [D1] şeklinde alıntıla.
"""
COMPETING_HYPOTHESES_EVALUATION_PROMPT = """
Sen Baş Ekonometrist olarak kanıtların sınırlarını değerlendir.
H0: {h0}; H1: {h1}; H2: {h2}. Kanıtlar: {evidence}
{domain_rules}
Her hipotez için kanıt ve sınırlama tablosu yaz. Hesaplanmış test yoksa betimsel değerlendirme yap.
Destek yüzdesi, p-değeri veya kazanan hipotez uydurma. Türkçe yanıtla.
"""
PREDICTIVE_INSIGHT_PROMPT = """
Sen bir Büyüme Stratejistisin. Konu: {topic}; dönemsel kanıtlar: {evidence}.
Yalnızca gözlenen dönemlerin yönünü ve nitel senaryoları anlat.
Sayısal tahmin, güven aralığı veya kanıtlanmamış nedensellik uydurma. Türkçe yanıtla.
"""
