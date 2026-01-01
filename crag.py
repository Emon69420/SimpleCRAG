from classes import Loader, Chunking, VectorStore, Retriever
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

url = [
    "https://ryanocm.substack.com/p/mystery-gift-box-049-law-1-fill-your",
    "https://ryanocm.substack.com/p/105-the-bagel-method-in-relationships",
    "https://ryanocm.substack.com/p/098-i-have-read-100-productivity"
]


## Classes called for LOADING -> CHUNKING -> VECTORIZATION -> RETRIEVAL
loading = Loader(url)
documents = loading.Load()
chunks = Chunking(documents)
chunked = chunks.Chunker()
vectorstore = VectorStore(chunked)
query = "bagel method in relationships"
vectors = vectorstore.store()
retriever = Retriever(vectors, query)
docs = retriever.retrieve(vectors)

#LLM for using the retrieved documents to answer the query
rag_prompt = hub.pull("rlm/rag-prompt") # Pre-built RAG prompt template
llm = init_chat_model(
    "google_genai:models/gemini-flash-lite-latest",
    temperature=0,
    api_key=""
)
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs) #joins all documents with double new lines

rag_chain = rag_prompt | llm | StrOutputParser() #Langchain language that says first use rag_prompt, then llm, then parse output as string
print(rag_prompt.messages[0].prompt.template)


generation = rag_chain.invoke({
    "context": format_docs(docs),
    "question": query
})

# Print the results
print("Question: %s" % query)
print("----")
print("Documents:\\n")
print('\\n\\n'.join(['- %s' % x.page_content for x in docs]))
print("----")
print("Final answer: %s" % generation)

