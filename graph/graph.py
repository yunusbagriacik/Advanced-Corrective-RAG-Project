#Bu dosya bütün node’ları bağlar.

from dotenv import load_dotenv

from langgraph.graph import END, StateGraph #LangGraph’ta graph oluşturmak için kullanılır.END, graph’ın bittiği noktayı temsil eder.

#Graph içinde karar vermek için kullanılan chain’leri import ediyoruz.
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

from graph.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH #Node isimleri import ediliyor.
from graph.nodes import generate, grade_documents, retrieve, web_search #Node fonksiyonları import ediliyor.
from graph.state import GraphState #State şeması import ediliyor.

load_dotenv()


def decide_to_generate(state):
    """
    Bu fonksiyon grade_documents node’undan sonra çalışıyor.

        Amacı:

        Dokümanlar yeterli mi?
        Yoksa web search gerekli mi?"""
    print("---ASSESS GRADED DOCUMENTS---")
    #Eğer web_search=True ise web search node’una gider. Değilse generate node’una gider.
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """Bu fonksiyon generate sonrası kalite kontrol yapar."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    #Cevap dokümanlara dayanıyor mu?
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    #Eğer cevap dokümanlara dayanıyorsa ikinci kontrole geçiyor.
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        #Cevap soruyu gerçekten cevaplıyor mu? Cevap soruyu cevaplıyorsa graph bitecek.Cevap soruyu cevaplamıyorsa web search’e gidecek.
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

#Graph’ın giriş kararıdır.
def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question}) #Router chain’e soru gönderilir.
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")