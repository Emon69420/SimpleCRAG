from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class Loader():
    def __init__(self, url):
        self.url: list = url

    def Load(self):
        data = WebBaseLoader(self.url)
        extracted = data.load()
        return extracted
    
class Chunking():

    textsplitter =  RecursiveCharacterTextSplitter(chunk_size = 128, chunk_overlap = 20 )

    def __init__(self, document):
        self.document = document

    def Chunker(self):
        chunks = self.textsplitter.split_documents(self.document)
        return chunks
    
class Embedding():
    
    def __init__(self, chunks):
        pass
    
    def Embedder(self):
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={"device": "cpu"}, 
            model_name = "BAAI/bge-base-en-v1.5",
        )
        return embeddings

class VectorStore(Embedding):

    def __init__(self, documents):
        self.documents = documents

    def store(self):
        
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={"device": "cpu"}, 
            model_name = "BAAI/bge-base-en-v1.5",
        )

        vectorDB = Chroma.from_documents(self.documents, embeddings)
        return vectorDB
    
