from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import EMBEDDING_MODEL_NAME, RETRIEVER_TOP_K

class R3Retriever:
    def __init__(self, raw_documents):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = FAISS.from_texts(raw_documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    def search(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        formatted_docs = [f"<doc>{doc.page_content[:512]}</doc>" for doc in docs]
        return "\n".join(formatted_docs)