from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v2 import BaseModel, Field
from langchain.chat_models import init_chat_model

class Evaluator(BaseModel):

    binary_score: str = Field(
        description="Documents are relevaant to the question, Yes or No"
    )

        
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



