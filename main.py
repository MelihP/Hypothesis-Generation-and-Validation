import pandas as pd
import ast
import json
from sqlalchemy import create_engine

# Veritabanı motorunu oluştur (Yerel bir SQLite dosyası)
engine = create_engine('sqlite:///insight_generation_bot.db')

# CSV'leri yükle
df_tweets = pd.read_csv('twitter_tweets_demo-brand.csv')
df_users_mongo = pd.read_csv('twitter_users_demo-brand.csv') # MongoDB formatlı karmaşık user datası
df_predictions = pd.read_csv('demo_brand_predictions.csv')
df_users_sql = pd.read_csv('demo_brand_users.csv') # Daha temiz, flat (düz) SQL formatı



# Eğer temiz SQL formatını kullanırsak (LLM için en ideali):
df_demographics = df_users_sql[['id', 'gender', 'age_range', 'location']].copy()

# Sütun isimlerini LLM'in kolay anlayacağı (verbose) şekilde yeniden adlandıralım
df_demographics.rename(columns={
    'id': 'user_id',
    'age_range': 'age_group',
    'location': 'user_location'
}, inplace=True)

# Eksik verileri temizle
df_demographics.dropna(subset=['age_group', 'gender'], inplace=True)



# String olarak gelen listeleri (Örn: "['demo-brand', 'arcelik']") Python listesine çeviren yardımcı fonksiyon
def parse_category(val):
    try:
        # String'i listeye çevir, eğer ilk elemanı almak istersen [0] ekle
        return ast.literal_eval(val)[0] 
    except:
        return None

# 1. Emotion Analysis (Duygu Analizi) Tablosunun Oluşturulması
df_emotion_raw = df_predictions[df_predictions['task_name'] == 'emotion'].copy()
df_emotion_raw['dominant_emotion'] = df_emotion_raw['category_value'].apply(parse_category)

df_emotion_analysis = df_emotion_raw[['tweet_id', 'author_id', 'dominant_emotion', 'prediction_month']].copy()


# 2. Consumer Journey (Tüketici Yolculuğu) Tablosunun Oluşturulması
df_journey_raw = df_predictions[df_predictions['task_name'] == 'consumer_journey'].copy()
df_journey_raw['journey_stage'] = df_journey_raw['category_value'].apply(parse_category)

df_consumer_journey = df_journey_raw[['tweet_id', 'author_id', 'journey_stage', 'prediction_month']].copy()


# 3. Trending Topics Tablosunun Oluşturulması
# Eğer task_name topic ise filtrele
df_topic_raw = df_predictions[df_predictions['task_name'].str.contains('topic', na=False)].copy()
df_topic_raw['topic_name'] = df_topic_raw['category_value'].apply(parse_category)

df_trending_topics = df_topic_raw[['tweet_id', 'topic_name', 'prediction_month']].copy()



# DataFrame'leri SQL tabloları olarak kaydet
# LLM için tablo isimlerinin net olması şarttır
df_demographics.to_sql('demographics', con=engine, index=False, if_exists='replace')
df_emotion_analysis.to_sql('emotion_analysis', con=engine, index=False, if_exists='replace')
df_consumer_journey.to_sql('consumer_journey', con=engine, index=False, if_exists='replace')
df_trending_topics.to_sql('trending_topics', con=engine, index=False, if_exists='replace')

print("Veritabanı başarıyla oluşturuldu ve tablolar eklendi!")



# Demografi ve Duygu tablolarını user_id (author_id) üzerinden birleştir
df_merged = pd.merge(df_emotion_analysis, df_demographics, left_on='author_id', right_on='user_id')

# Yaş grubuna göre dominant duyguların yüzdesini hesapla
emotion_by_age = df_merged.groupby(['age_group', 'dominant_emotion']).size().reset_index(name='volume')
emotion_by_age['demographic_percentage'] = emotion_by_age.groupby('age_group')['volume'].transform(lambda x: (x / x.sum()) * 100)

# Bunu doğrudan LLM için optimize edilmiş özel bir tablo olarak SQL'e kaydet
emotion_by_age.to_sql('emotions_by_age_groups', con=engine, index=False, if_exists='replace')