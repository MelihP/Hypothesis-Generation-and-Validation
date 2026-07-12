import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# Groq API Anahtarını Buraya Yapıştır
os.environ["GROQ_API_KEY"] = ""

db = SQLDatabase.from_uri("sqlite:///insight_generation_bot.db")

# LLM Motorunu Groq (Llama 3 70B) olarak ayağa kaldır
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)

test_sorusu = "18-29 yaş grubundaki kullanıcılar veritabanında toplam kaç kişi? Bana net sayıyı ver."
print(f"\nSoru: {test_sorusu}\n")

response = agent_executor.invoke({"input": test_sorusu})
print("\n--- AJANIN CEVABI ---")
print(response["output"])