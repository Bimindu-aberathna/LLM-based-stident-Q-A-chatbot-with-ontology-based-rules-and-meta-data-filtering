from app.abstract_factory.Embedder.embedder import Embedder
import getpass
import os
from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder(Embedder):
    def __init__(self):
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        try:
            embeddings = self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Error embedding documents: {e}")

        for (i, embedding) in enumerate(embeddings):
            print(f"Document {i} embedding: {embedding}")

        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []