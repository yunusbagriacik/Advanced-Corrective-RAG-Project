#Bu dosya getirilen dokümanların soruyla alakalı olup olmadığını kontrol eder.
#Bu doküman soruyla alakalı mı? yes / no

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from ingestion import retriever
load_dotenv()

llm = ChatOpenAI(temperature=0) #yaratıcı olmasını istemediğimiz için 0 verdik.


class GradeDocuments(BaseModel): #Structured output modeli.
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments) #LLM çıktısını GradeDocuments formatında döndürür.

#LLM’e görev veriliyor:
#Retrieved document soruyla alakalı mı?Keyword veya semantic meaning varsa yes de.Yoksa no de.
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

"""
chain:
document + question
↓
prompt
↓
LLM
↓
GradeDocuments(binary_score="yes/no")
"""

"""
if __name__ == "__main__":
    user_question = "What is fireflies?"
    docs = retriever.get_relevant_documents(user_question)
    retrieved_document = docs[0].page_content
    print(retrieval_grader.invoke({"question": user_question, "document": retrieved_document}))
    
"""