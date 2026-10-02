# Pazarlama Veri ve İçgörü Motoru

Streamlit, SQLite, LangChain ve OpenAI ile beş analiz akışı:

- **Otonom içgörü:** Mevcut şemaya uygun araştırma sorusu ve kanıtlı özet.
- **Manuel soru:** SQL, belge veya hibrit yönlendirme; yapılandırılmış filtre hafızası ve netleştirme.
- **Hipotez değerlendirme:** H0/H1/H2 için betimsel kanıt tablosu. İstatistiksel test olmayan bir rapor kesin doğrulama veya destek yüzdesi üretmez.
- **Trend senaryosu:** Birden fazla veri dönemi gerektirir. Sayısal tahmin modeli değildir.
- **Veri Sorgulama ve İstatistik:** Anahtarsız tablo oluşturucu, doğal dil sorgusu, çapraz tablolar, sayım/paylar, CSV/Excel indirme, tam tablo NULL analizi ve hesaplanmış demografik ilişki testleri.

## Kurulum

Python 3.12 ile:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c requirements.lock
python -m pip check
python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
```

Bulut ortamında `/workspace/hypothesis-venv` kullanılabilir. Her görev zaten izoledir; açıkça istenmedikçe yeni Git worktree oluşturmayın.

Doğal dil ve yorumlama için güvenli ortam ayarlarında `OPENAI_API_KEY` veya `LLM_API_KEY` bağlayın; `LLM_API_KEY` uygulama içinde desteklenir. Streamlit secrets da kullanılabilir. Anahtarları kaynak koduna, loglara veya Git'e eklemeyin. Tablo oluşturma, veri kalite ve istatistik araçları OpenAI gerektirmez.

## Belge araması

`PINECONE_API_KEY` olmadan SQL araçları çalışır; belge soruları açıklayıcı hata verir. Pinecone bağlantısı yalnızca belge/hibrit analiz gerektiğinde açılır.

- `PINECONE_INDEX`: Varsayılan `pazarlama-verileri`.
- `PINECONE_NAMESPACE`: Varsayılan boş namespace; kuruma ait belge alanını belirtmek için kullanılabilir.
- Embedding modeli: `text-embedding-3-small`; indeks boyutu varsayılan embedding boyutuyla uyumlu olmalıdır.
- Belgeler indekslenmiş olmalıdır; bu uygulama PDF yükleme/indeks oluşturma hattı içermez.
- İndeks metadata'sında `source`/`file_name`, `page`/`section` ve `date` alanları önerilir. Eksik kaynak alanları uydurulmaz.
- Kısıtlı ağda `api.openai.com`, `api.pinecone.io` ve Pinecone indeksinizin gerçek HTTPS hostname'i erişilebilir olmalıdır. Proxy üzerinden anahtar kullanılıyorsa anahtar hedefleri de ilgili Pinecone hostname'lerini içermelidir.

Belge parçaları kanıt olarak kullanılır; yorumda `[D1]` biçiminde geçerli kaynak atfı gerekir. Kaynak tablosu dosya, sayfa/bölüm ve tarih bilgisini gösterir. Namespace ayarı tek başına çok kullanıcılı yetkilendirme sağlamaz; üretim yayımı için kullanıcı/kurum erişim katmanı ayrıca gerekir.

## Veri sözlüğü ve analiz sınırları

| Tablo | Analiz birimi / ilişkisi |
|---|---|
| `demographics` | Kullanıcı; anahtar `user_id`, `age_group`, `gender` |
| `consumer_journey` | Tweet yolculuk etiketi; `author_id = demographics.user_id` |
| `emotion_analysis` | Tweet duygu etiketi; aynı kullanıcı ilişkisi |
| `trending_topics` | Bir tweet birden fazla konu kaydı taşıyabilir |
| `emotions_by_age_groups` | Önceden hesaplanmış yaş/duygu hacimleri ve oranları |

Konu-yolculuk/duygu JOIN'lerinde `tweet_id` ve `prediction_month` birlikte eşleştirilir. Kayıt, farklı tweet ve farklı kullanıcı sayıları ayrı metriklerdir. `prediction_month` YYYYMM etiketidir; olay tarihi olduğu varsayılmaz. Mevcut depoda tarih içeren tablolarda yalnızca **202607** vardır. Olmayan dönem, ölçülmüş sıfır veya huni daralması değildir.

Görünen tablo paylarının paydası gösterilen grup hacimlerinin toplamıdır. Bunlar özellikle top-N, kesilmiş veya çok etiketli sonuçlarda nüfus oranı değildir. İstatistik özetleri gösterilen sonuç satırlarına aittir; gruplanmış hacimlerin ortalaması ham gözlem ortalaması değildir. Kullanıcı/tweet ID'leri ve dönem etiketlerinin ortalama/standart sapması hesaplanmaz.

### Hesaplanan istatistikler

Demografik karşılaştırma, seçili dönemde tek kaydı olan kullanıcılardan bir frekans tablosu oluşturur. Tekil demografik anahtar kontrolü yapılır; tekrarlı, eşleşmeyen veya eksik etiketli kullanıcılar dışlanır ve kapsam raporlanır.

- Uygun hücre hacminde ki-kare bağımsızlık testi.
- Seyrek 2×2 tabloda Fisher exact testi; daha büyük seyrek tabloda yetersiz kanıt.
- Bias düzeltmeli Cramér V etki büyüklüğü.
- Grup içi oranlar ve bireysel %95 Wilson güven aralıkları.
- 2×2 tabloda oran farkı ve Newcombe-Wilson %95 aralığı.

p-değeri hipotezin doğru olma olasılığı değildir. Testler keşifseldir; farklı analizlerdeki tekrarlar için otomatik çoklu test düzeltmesi veya temsilî örneklem garantisi yoktur. Sosyal medya ve tahmini demografik etiketlerden nedensellik veya satış dönüşümü çıkarılmaz.

## Güvenli yürütme ve kanıt kontrolü

SQLite dosyası salt okunur açılır. Değerler parametreyle bağlanır; tablo/sütunlar canlı şemada doğrulanır. Ham JOIN/SQL koşulları kabul edilmez. Yazma, ATTACH ve PRAGMA işlemleri yürütücüde reddedilir. Varsayılan 200, üst sınır 1.000 sonuç satırıdır; fazladan bir satır kontrolüyle kesilme görünür yapılır. En fazla 3 JOIN ve sınırlı alt sorgu derinliği vardır. Yürütme 5 saniye/20 milyon VM adımıyla sınırlıdır.

Boş, başarısız veya kesilmiş sonuçlar nihai yorum kanıtı sayılmaz. Kısmi yönetici özeti yalnızca başarılı kanıtlara dayanır ve açıkça etiketlenir. Hipotez/trend değerlendirmesi gerekli adımların tümü başarılı değilse üretilmez. Excel/CSV çıktılarında metin formülleri kaçışlanır.

## Doğrulama

```bash
python -m unittest discover -s tests -v
```

CI anahtarsız çalışır: derleyici, salt okunur yürütme, timeout, kanıt kapıları, kaynak yönlendirme/atıf, istatistik formülleri, indirmeler, sohbet netleştirmesi ve beş modun Streamlit akışları test edilir. RAG testleri örnek retriever kullanır; gerçek Pinecone erişimi anlamına gelmez.

OpenAI anahtarı bağlıyken, ücretli API çağrıları yapan isteğe bağlı kontrol:

```bash
python scripts/smoke_openai.py
```

Bu kontrol gerçek model planını doğrudan SQL toplamıyla karşılaştırır; otomatik SQL yönlendirmesini, kanıtlı özeti ve dönem bilgisinin takip sorusunda korunmasını sınar. Canlı model davranışı değişebileceği için CI'ye dahil değildir.
