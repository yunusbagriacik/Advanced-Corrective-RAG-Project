from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter #Metinleri küçük parçalara bölmek için kullanılır. Çünkü RAG sistemlerinde bütün blog yazısını tek parça olarak modele vermek doğru değildir.
from langchain_community.document_loaders import WebBaseLoader #Web sayfasından içerik çekmek için kullanılır.
from langchain_community.vectorstores import Chroma #Chroma vector database’i kullanmak için import edilir.
from langchain_openai import OpenAIEmbeddings #Metni vektöre çevirmek için OpenAI embedding modeli kullanılır.

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls] #Her URL için web sayfasını yükler.#
# Yani liste içinde liste gelir.
"""
[
    [Document(page_content="...", metadata={"source": "url1"})],
    [Document(page_content="...", metadata={"source": "url2"})],
    [Document(page_content="...", metadata={"source": "url3"})],
]
"""
docs_list = [item for sublist in docs for item in sublist] #Listeyi düzleştirir. [[doc1], [doc2], [doc3]] -> [doc1, doc2, doc3]

#Bu dosyayı doğrudan çalıştırırsan doküman listesini yazdırır. #
#Ama başka dosyadan import edilirse bu print çalışmaz.
if __name__== "__main__":
    print(docs_list)

#Dokümanları token bazlı bölecek splitter oluşturur.Her chunk yaklaşık 250 token olsun.
#chunk_overlap=0-> Chunk’lar arasında tekrar eden metin olmasın.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
#Büyük dokümanları küçük parçacıklara böler.Örn: 3 blog yazısı → 100+ küçük chunk
doc_splits = text_splitter.split_documents(docs_list)

"""
doc_splits içindeki her chunk’ı al
↓
OpenAIEmbeddings ile vektöre çevir
↓
rag-chroma collection içine kaydet
↓
.chroma klasöründe diske yaz

Yani aşağıdaki satırdan sonra local vector database oluşuyor.
"""
vectorstore = Chroma.from_documents(
     documents=doc_splits,
     collection_name="rag-chroma",
     embedding=OpenAIEmbeddings(),
     persist_directory="./.chroma",
 )


"""
Var olan Chroma database’i açar ve retriever’a çevirir.

Retriever’ın görevi:

Soru ver → en alakalı doküman chunk’larını getir

Örnek:

retriever.invoke("What is adversarial attack?")
"""
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

