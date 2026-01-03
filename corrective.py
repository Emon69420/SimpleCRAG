from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Rewriter():
    def __init__(self, api_key, query):
        self.api_key: str = api_key
        self.query: str = query

    def rewrite(self):
        llm = init_chat_model(
            "google_genai:models/gemini-flash-lite-latest",
            temperature=0,
            api_key=self.api_key
        )
        system = """You are a question re-writer that converts an input question to a better version optimized for web search.
        Rewrite the user's question as a single improved search query. Do not include explanations, options, or reasoning. Only output the rewritten question."""
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        question_writer = rewrite_prompt | llm | StrOutputParser()
        
        return question_writer.invoke({"query": self.query})