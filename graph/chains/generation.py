from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0) #yaratıcı olmasını istemediğimiz için 0 verdik.
prompt = hub.pull("rlm/rag-prompt") #hubtaki promptu çek

generation_chain = prompt | llm | StrOutputParser() #chain: promptu al -> modele ver -> parser et