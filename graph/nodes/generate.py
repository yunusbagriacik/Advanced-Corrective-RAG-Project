#Bu node cevap üretir.
from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    #Soru ve context dokümanlarını alır.
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question}) #RAG chain’i çalıştırır.
    return {"documents": documents, "question": question, "generation": generation} #State’e üretilen cevabı ekler.

"""
context = retrieved / web searched documents
question = user question
↓
LLM answer
"""