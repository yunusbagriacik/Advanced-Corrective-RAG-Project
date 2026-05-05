#Bu dosya graph boyunca taşınacak state yapısını tanımlar.
from typing import List, TypedDict #TypedDict, dictionary’nin hangi key’lere sahip olacağını belirtmek için kullanılır.

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]