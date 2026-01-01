from langchain.chat_models import init_chat_model
from langchain.prompts.chat import ChatPromptTemplate
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
        system = """You are a question re-writer that converts an input question to a better version that is optimized \\n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Rewrite the following question for web search, Formulate an improved question.: {query}"),
        ])
        question_writer = rewrite_prompt | llm | StrOutputParser()
        
        return question_writer.invoke({"query": self.query})