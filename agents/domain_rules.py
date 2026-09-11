"""
Pazarlama Alan Bilgisi ve Kavramsal Sözlük Katmanı (Domain Rules & Business Logic Layer).
Botun SQL çıktılarını mekanik olarak değil; pazarlama hunisi, nedensellik ve marka sağlığı
kavramlarıyla stratejik bir analist gibi yorumlamasını sağlar.
"""

from typing import Dict, Any


# --- 1. KAVRAMSAL SÖZLÜK (CONCEPTUALIZATION DICTIONARY) ---
MARKETING_CONCEPT_DEFINITIONS = """
1. CONSUMER JOURNEY (TÜKETİCİ HUNİSİ AŞAMALARI):
   - Consideration (Değerlendirme - Top of Funnel): Tüketicinin markayı alternatifler arasına alması, merak etmesi ve satın alma niyetidir. Hacmin düşmesi veya sıfıra yaklaşması "veri yokluğu" DEĞİL, "huninin tepe noktasının daralması" (top-of-funnel narrowing) ve yeni müşteri akışının kesilmesi anlamına gelir.
   - Purchase (Satın Alma - Middle/Bottom of Funnel): Satın alma eylemi, sipariş ve işlem deneyimidir.
   - Recommendation (Tavsiye / Savunuculuk - Retention & Advocacy): Memnuniyet sonrası başkalarına önerme ve organik savunuculuk durumudur.
   - Complaint (Şikayet - Friction Point): Satış öncesi veya sonrası yaşanan sürtünmeler, arızalar ve operasyonel aksaklıklardır.

2. BRAND HEALTH VE NEDENSELLİK BAĞLAMLARI:
   - Solution (Çözüm Algısı): Tüketicinin tavsiye etmese (düşük Recommendation) bile pratik fayda veya mecburiyetten markayı satın almaya devam etmesi durumunu açıklar.
   - Reason to Believe (İkna Edici Nedenler): Tüketicinin markaya güvenme dayanaklarıdır.
   - Demografik Kırılım: Yaş ve cinsiyet gruplarının farklı aşamalarda gösterdiği ayrışmalardır (Örn: 40+ yaşta müşteri hizmetleri kaynaklı şikayet yoğunlaşması).
"""


# --- 2. STRATEJİK YORUMLAMA KURALLARI (HEURISTIC RULES) ---
BUSINESS_HEURISTIC_RULES = """
STRATEJİK ÇIKARIM VE HİPOTEZ DEĞERLENDİRME KURALLARI:

KURAL 1: VERİ YOKLUĞUNU STRATEJİK YORUMLA (NULL/ZERO HANDLING)
- 'Consideration' aşamasında bir dönemde (örneğin 2026) tweet sayısı 0 veya çok düşükse, ASLA "Veri bulunamadı, hipotez çürütüldü" deme.
- Doğru Yorum: "Consideration hacminin sıfırlanması/çökmesi, pazarlama hunisinin tepe noktasının ciddi biçimde daraldığını (top-of-funnel narrowing) ve markanın yeni tüketici çekme gücünü kaybettiğini kanıtlamaktadır."

KURAL 2: DİVERJANSLARI (ÇELİŞKİLİ TRENDLERİ) ÇÖZÜMLE
- Eğer 'Recommendation' düşerken 'Purchase' sabit kalıyor veya artıyorsa: Bunu çelişki olarak adlandırma. "Tüketicilerin tavsiye motivasyonu kırılmış olsa da markayı vazgeçilmez veya pratik bir 'Solution' (Çözüm) olarak görmeye devam ettikleri" hipotezini kur.

KURAL 3: BİLGİ DÜZEYİ VE İSPAT DERECELENDİRMESİ
Her hipotez veya içgörü sonucunda iddiayı şu 3 seviyeden biriyle etiketle:
- [Doğrulandı / Demonstrated]: Doğrudan SQL verisiyle birebir ispatlanan durumlar (Örn: "Şikayetlerin %65'i kargo gecikmesinden kaynaklanmaktadır").
- [Makul / Plausible]: Trendlerin desteklediği ancak doğrudan nedensellik ispatlanamayan durumlar (Örn: "Düşüş dönemsel/konjonktürel bir etkiden kaynaklanıyor olabilir").
- [Desteklenmeyen / Unsupported]: Veri setinde doğrudan sütunu olmayan kavramlarla yapılan aşırı yorumlar (Örn: Elde tekrar eden alım veya sadakat tablosu yokken "Sadık müşteriler sayesinde ayakta duruyor" demek).
"""


# --- 3. PROMPT ENTEGRASYON ŞABLONLARI ---

def get_domain_context_prompt() -> str:
    """Ajanın sentez ve doğrulama promptlarına eklenecek kurumsal alan bilgisi metni."""
    return f"""
AŞAĞIDAKİ PAZARLAMA VE STRATEJİ KURALLARINA KESİNLİKLE UY:
{MARKETING_CONCEPT_DEFINITIONS}

{BUSINESS_HEURISTIC_RULES}
"""


def build_hypothesis_synthesis_prompt(hypothesis: str, sql_evidence: str) -> str:
    """
    Hipotez doğrulama aşamasında LLM'e verilecek nihai değerlendirme talimatı.
    """
    domain_rules = get_domain_context_prompt()
    return f"""\
Sen üst düzey bir Pazarlama ve Büyüme Direktörüsün (CMO).
Kullanıcının sınamak istediği Hipotez: "{hypothesis}"

Veritabanından toplanan SQL Kanıtları:
{sql_evidence}

{domain_rules}

GÖREVİN:
1. SQL kanıtlarını pazarlama kuralları çerçevesinde analiz et.
2. Hipotezin durumunu net olarak belirt: [KABUL / DOĞRULANDI], [MAKUL / KISMEN DESTEKLENDİ] veya [REDDEDİLDİ].
3. Eğer veri 0 veya eksikse, bunun funnel daralması veya tüketici ilgisizliği anlamına gelip gelmediğini açıkla.
4. Yöneticiye 3 maddelik somut ve aksiyonel stratejik tavsiye sun.

Yanıtını profesyonel, net ve analitik bir dille Türkçe olarak hazırla.
"""