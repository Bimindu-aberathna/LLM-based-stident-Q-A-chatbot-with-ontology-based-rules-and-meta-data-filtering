from pydantic import BaseModel
from typing import Optional

class EvaluationLog(BaseModel):
    system_answer_text: str = ""
    retrieved_doc_names_ordered: str = ""
    retrieved_chunks_text: str = ""
    _temp_chunks: list = []  # Private field to store chunks temporarily
    _temp_doc_names: list = []  # Private field to store doc names temporarily