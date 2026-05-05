#Bu dosya cevap üretme chain’idir.

from langchain import hub #LangChain Hub’dan hazır prompt çekmek için kullanılır.
from langchain_core.output_parsers import StrOutputParser #LLM cevabını string’e çevirmek için kullanılır.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0) #yaratıcı olmasını istemediğimiz için 0 verdik.
prompt = hub.pull("rlm/rag-prompt") #hubtaki promptu çek

generation_chain = prompt | llm | StrOutputParser() #chain: promptu al -> modele ver -> parser et



"""
rlm/rag-prompt:
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


"""
context + question
↓
RAG prompt
↓
LLM
↓
string cevap
"""