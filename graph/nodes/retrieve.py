#Bu graph node’u retrieval yapar.

from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

#LangGraph node fonksiyonu.
def retrieve(state: GraphState) -> Dict[str, Any]: #Input olarak state alır, dictionary döndürür.
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question) #Vectorstore’dan soruya en alakalı chunk’ları getirir.
    return {"documents": documents, "question": question} #State’i günceller.