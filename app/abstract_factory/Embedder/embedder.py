from abc import ABC, abstractmethod
class Embedder(ABC):
    @abstractmethod
    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        pass
    
    def embed_query(self, text: str) -> list[float]:
        pass