from typing import Literal #verdiğimiz yapının herhangi birini seçme özelliği için import edildi.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., #üç nokta konulmasının sebebi ya vectorstore ya da websearch almak zorundasın demek.
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0) #yaratıcı olmasını istemediğimiz için 0 verdik.
structured_llm_router = llm.with_structured_output(RouteQuery) #llm çıktısını yapısal bir şekilde ele al ve llm ile bağla.

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router # route_promptu al -> structured_llm_router'a ver.

"""
if __name__ =="__main__":
    print(question_router.invoke({"question" : "What is Docker?"}))
"""
