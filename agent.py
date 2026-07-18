import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# Hatalı eski kütüphane yerine, doğrudan ana çekirdek aracı (Tool) çağırıyoruz
from langchain_core.tools import Tool

# --- 1. API ANAHTARLARI (GÜVENLİK BÖLGESİ) ---
os.environ["OPENAI_API_KEY"] = "xxxxx"
os.environ["PINECONE_API_KEY"] = "xxxxx"


def get_hybrid_agent():
    # --- 2. SQL BAĞLANTISI (Sayılar ve Tablolar için) ---
    db = SQLDatabase.from_uri("sqlite:///insight_generation_bot.db")
    
    # --- 3. YAPAY ZEKA BEYNİ VE VEKTÖR MOTORU ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- 4. RAG BAĞLANTISI (Metinler ve PDF'ler için) ---
    # Pinecone'daki index adını buraya yazmayı unutma
    index_name = "pazarlama-verileri" 
    
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # IMPORT HATASINI KÖKÜNDEN ÇÖZEN YENİ KOD BLOĞU:
        # Aracı dışarıdan çağırmak yerine, Tool sınıfıyla sıfırdan kendimiz yaratıyoruz
        rag_tool = Tool(
            name="dokuman_arama_araci",
            description="Markanın iade politikaları, PDF raporları veya SQL veritabanında OLMAYAN yapılandırılmamış (unstructured) metinleri araştırmak için bu aracı kullan.",
            func=retriever.invoke
        )
        ekstra_araclar = [rag_tool]
        print("✅ RAG Aracı başarıyla sisteme entegre edildi!")
    except Exception as e:
        print(f"⚠️ RAG sistemine bağlanılamadı, sadece SQL aracı devrede. Hata: {e}")
        ekstra_araclar = []

    # --- 5. HİBRİT AJAN OLUŞTURMA (SQL + RAG) ---
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        extra_tools=ekstra_araclar,
        verbose=False
    )
    
    return db, llm, agent_executor

db, llm, agent_executor = get_hybrid_agent()