from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END #framework helps to design and manage flow of tasks in an application
from classes import Loader, Chunking, VectorStore, Retriever
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from corrective import Rewriter

api_key = ''

# This is just an evaluator class to give binary score on whether the document is relevant to the query
#Not part of the graph
class Evaluator(BaseModel):

    binary_score: str = Field(
        description="Documents are relevaant to the question, Yes or No"
    )

class AgentState(TypedDict):
    query: str
    documents: List[str]
    websearch: str
    generation:str

def retrieve(state: AgentState) -> AgentState:

    """Retrieves documents based on the query in the state."""

    query = state["query"]
    url = [
    "https://ryanocm.substack.com/p/mystery-gift-box-049-law-1-fill-your",
    "https://ryanocm.substack.com/p/105-the-bagel-method-in-relationships",
    "https://ryanocm.substack.com/p/098-i-have-read-100-productivity"
]
    loading = Loader(url)
    documents = loading.Load()
    chunks = Chunking(documents)
    chunked = chunks.Chunker()
    vectorstore = VectorStore(chunked)
    vectors = vectorstore.store()
    retriever = Retriever(vectors, query)
    docs = retriever.retrieve(vectors)

    return {"documents": docs, "query": query}

def evaluate(state: AgentState) -> AgentState:
    """Evaluates the relevancy of retrieved documents to the query."""

    question = state["query"]
    documents = state["documents"]
    api = api_key
    llm = init_chat_model(
    "google_genai:models/gemini-flash-lite-latest", 
    temperature=0,
        api_key=api
    )
    structured_llm = llm.with_structured_output(Evaluator)
    system = """You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question. \\n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \\n
    Output a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question} \\n Document: {document} \\n"),
    ])

    eval_grader = evaluator_prompt | structured_llm
    for doc in documents:
        result = eval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        if result.binary_score.lower() == 'yes':
            state["generation"] = "yes"
        else:
            state["generation"] = "no"

        return state
    
def corrective(state: AgentState) -> AgentState:
    """Correction Of Prompt for searching the web"""
    correction = Rewriter(api_key, state["query"])
    rewritten_query = correction.rewrite()
    state["query"] = rewritten_query
    print("Rewritten Query:", rewritten_query)
    return state


def decide(state: AgentState) -> AgentState:
    """Deciding Next Step Based on Evaluation."""

    if state["generation"] == "no":
        return "corrective"

    elif state["generation"] == "yes":
        return "generation"
    

            

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("evaluate", evaluate)
graph.add_node("router", lambda state:state)
graph.add_node("corrective", corrective)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "evaluate")
graph.add_conditional_edges("router",
                            decide,
                            {
                                "corrective":"corrective",
                                "generation":END
                            }
                            )
graph.add_edge("evaluate", "router")
graph.add_edge("corrective", END)

app = graph.compile()
result = app.invoke({"query": "Do you know about that gun law in texas like what do i need to have legally?"})
print(result)

