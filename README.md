# AI-Powered Hybrid Marketing Insight Engine

ClickHouse ve SQLite verilerini, gerektiğinde Pinecone dokümanlarıyla birleştiren Streamlit analiz uygulaması.

## Çalışma modları

- **Otonom İçgörü:** Canlı şemaya göre sorular üretir; SQL, doküman veya hibrit arama kullanır.
- **Manuel Soru:** Belirsiz soruları netleştirir ve sınırlı konuşma geçmişiyle takip sorularını çözer.
- **Yarışan Hipotezler:** Kanıtları karşılaştırır. Eksik, başarısız veya kesilmiş sorgulardan nihai karar üretmez; istatistiksel test yapılmadan hipotezi kanıtlanmış saymaz.
- **Tahminleme:** Dönem kapsamını kontrol eder. Tek dönemden gelecek trendi üretmez; çıktı keşifseldir.
- **📊 Veri Sorgulama ve İstatistik:** Doğal dilde veya alan seçerek tablo, sayım, farklı kullanıcı/tweet sayısı, çapraz tablo, grafik ve CSV/Excel indirme. Tam tablo üzerinde NULL analizi; bağımsız birimler için ki-kare/Fisher, Cramér V, oranlar ve %95 güven aralıkları.

Model profilleri **Hibrit**, **Tasarruf**, **Hassas** ve **Özel** üzerinden seçilir. SQL ve tablo oluşturma için Pinecone gerekli değildir. Doküman araması yalnızca gerektiğinde başlatılır; kaynaklar `[D1]` biçiminde doğrulanır. Sonuç tabloları filtreleri, analiz birimini, kesilme durumunu ve sorgu hatalarını gösterir.

## Kurulum

Python 3.12 ile doğrulanmıştır:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
streamlit run app.py
```

`requirements.txt` doğrudan bağımlılıkları, `requirements.lock` doğrulanmış sürümleri içerir. Streamlit Community Cloud giriş dosyası **app.py** olarak kalır; mevcut ortamınızın Python sürümünü 3.12 seçin.

`.streamlit/secrets.toml.example` dosyasını temel alın. Gerçek anahtarları Git'e eklemeyin. Streamlit Cloud uygulamasının **Settings → Secrets** bölümünde mevcut adlar korunur:

```toml
OPENAI_API_KEY = "your-key"
CLICKHOUSE_HOST = "your-clickhouse-host"
CLICKHOUSE_PORT = "8443"
CLICKHOUSE_USERNAME = "read-only-user"
CLICKHOUSE_PASSWORD = "your-password"
CLICKHOUSE_DB = "your-database"
CLICKHOUSE_SECURE = "true"
CLICKHOUSE_VERIFY = "true"
ALLOW_SQLITE_FALLBACK = "false"
```

`LLM_API_KEY`, `OPENAI_API_KEY` yerine de kullanılabilir. Pinecone modu için `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` ve isteğe bağlı `PINECONE_NAMESPACE` kullanılır. Özel TLS sertifikaları için `CLICKHOUSE_CA_CERT` ayarlanabilir. Mevcut `CLICKHOUSE_VERIFY=false` yapılandırması korunur ve arayüzde uyarılır; üretimde doğrulamanın açık olması önerilir. Port 443/8443 varsayılan olarak TLS kullanır.

ClickHouse bağlantısı tanımlı değilse yerel `SQLITE_DB_PATH` (varsayılan `insight_generation_bot.db`) kullanılır. Tanımlı ClickHouse bağlantısı başarısızsa uygulama hata gösterir; **sessizce başka veri kümesine geçmez**. SQLite yedeklemesi yalnızca `ALLOW_SQLITE_FALLBACK=true` ile açılır ve arayüzde açıkça belirtilir. Etkin veri kaynağı her zaman görünür.

## ClickHouse ve istatistik kapsamı

Canlı kolonlar `tweet_predictions`, `tweets`, `users`, `user_factors` tablolarından okunur. Var olmayan kolon/ilişkiler uydurulmaz. `Array(String)`, `has/hasAny/hasAll`, `groupArray` ve JOIN sonrası `ARRAY JOIN` desteklenir. Dizi gruplamasında **Dizi etiketlerini ayrı satırlara aç** seçeneği kullanılır; aynı birimin birden fazla grupta görünebileceği uyarılır. Tarih alanları aylık kapsam seçimine çevrilir.

İstatistiksel karşılaştırmada bağımsız analiz anahtarı, etiket, grup tablosu ve eşleşme anahtarı seçilir. Aynı birime ait çoklu kayıtlar, çoklu etiketler, çoklu JOIN eşleşmeleri ve eksik değerler dışlanır; dışlanan birim sayısı gösterilir. Seyrek 2×2 tabloda Fisher testi uygulanır; daha büyük seyrek tablolarda güvenilir test sonucu üretilmez. Güven aralıkları çoklu karşılaştırma düzeltmesi içermez; sonuç nedensellik kanıtı değildir.

Sorgular yalnızca okuma içindir. Parametreler driver üzerinden bağlanır; tablo/kolonlar şemaya göre doğrulanır. Sonuç sınırı en fazla 1.000 satırdır. ClickHouse sorguları 5 saniye, 256 MiB bellek ve 10 milyon okunan satırla sınırlandırılır; büyük sorgular kapsam daraltılmasını gerektirebilir. Üretimde ayrıca yalnızca gerekli tablolara SELECT yetkili kullanıcı kullanın. CSV/Excel çıktıları formül enjeksiyonuna karşı korunur.

## Testler

```bash
python -m unittest discover -s tests -v
```

Bu testler gerçek OpenAI/Pinecone anahtarı gerektirmez. ClickHouse entegrasyon testleri yalnızca `127.0.0.1:18123` üzerindeki test sunucusuna bağlanır ve **hypothesis_fixture** veritabanını oluşturup siler. Üretim sunucusuna yönlendirmeyin.

```bash
docker run -d --name hypothesis-clickhouse-test --cpus=2 --memory=2g \
  -p 127.0.0.1:18123:8123 \
  -e CLICKHOUSE_USER=hypothesis_test -e CLICKHOUSE_PASSWORD=local-test-only \
  clickhouse/clickhouse-server:25.8-alpine
# Sunucu hazır olduktan sonra:
LOCAL_CLICKHOUSE_TEST=1 python -m unittest discover -s tests -v
docker rm -f hypothesis-clickhouse-test
```

GitHub Actions aynı testleri ayrı bir ClickHouse servisiyle çalıştırır. Canlı Streamlit güncellemesi için bu değişikliklerin uygulamanın izlediği dala (bu depoda `work`) birleştirilmesi ve uygulamanın yeniden başlatılması gerekir.

## Streamlit Secrets ve Pinecone uyumu

Anahtarlar bu uygulamanın Streamlit **Settings → Secrets** bölümünde veya çalışma ortamında tanımlanmalıdır. GitHub Actions Secrets, Streamlit uygulamasına otomatik aktarılmaz. Secrets güncellendikten sonra uygulamayı yeniden başlatın veya **Bağlantıları yeniden yükle** düğmesini kullanın; önceki veri kaynağının sohbet ve sonuçları temizlenir. Pinecone için arayüzdeki “anahtar tanımlı” bilgisi yalnızca yapılandırma varlığını gösterir, canlı bağlantı doğrulaması değildir.

Mevcut düz anahtarlar korunur. İsterseniz Streamlit TOML bölümleri de kullanılabilir:

```toml
[openai]
api_key = "your-openai-key"

[pinecone]
api_key = "your-pinecone-key"
index_name = "pazarlama-verileri"
namespace = ""
embedding_model = "text-embedding-3-small"
text_key = "text"

[clickhouse]
host = "your-host"
port = 8443
username = "readonly-user"
password = "your-password"
database = "your-database"
secure = true
verify = true
```

Düz Secrets değerleri bölüm ayarlarından, her ikisi ortam değişkenlerinden önce okunur. Pinecone'da index adı, namespace, metin metadata alanı (`PINECONE_TEXT_KEY`, varsayılan `text`) ve embedding modeli mevcut belgelerin yükleme ayarlarıyla aynı olmalıdır. Varsayılan model `text-embedding-3-small` olarak korunur. İsteğe bağlı `PINECONE_EMBEDDING_DIMENSIONS` boyutu index ile aynı olmalı; yalnızca text-embedding-3 modelleri özel boyutu destekler. Yanlış namespace boş sonuç, yanlış model/boyut başarısız veya alakasız arama üretebilir. Var olan belgeler bu değişiklikle yeniden yüklenmez.

**Otomatik otonom analiz**, sorusu veritabanı şemasından üretildiği için SQL kullanır. Pinecone'u otonom modda da kullanmak için **Belgeler** veya **Veritabanı + Belgeler** seçin. Manuel moddaki otomatik yönlendirme SQL/doküman/hibrit seçimini sürdürür. Belgelerden dönen kaynaklar ve atıflar doğrulanır; eksik kanıttan özet üretilmez.
