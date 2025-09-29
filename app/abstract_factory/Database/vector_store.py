from abc import ABC, abstractmethod
from typing import List, Optional, Dict

class VectorStore(ABC):
    def __init__(self):
        self.vectors = []
        
    @abstractmethod
    def store_vectors(self, vectors: List[List[float]], chunks: List[str], 
                     metadatas: Optional[List[Dict]] = None) -> None:
        pass

     