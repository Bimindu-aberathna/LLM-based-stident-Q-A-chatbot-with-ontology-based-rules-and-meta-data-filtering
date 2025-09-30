"""
Abstract LLM Response Generator

This module defines the abstract base class for LLM response generation services.
All LLM providers must implement the GenerateResponse method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMResponseGenerator(ABC):
    """
    Abstract base class for LLM response generators.
    
    This class defines the interface that all LLM providers must implement
    using the Abstract Factory design pattern.
    """
    
    @abstractmethod
    def GenerateResponse(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response using the LLM.
        
        Args:
            prompt (str): The input prompt/question
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Response containing generated text and metadata
        """
        pass