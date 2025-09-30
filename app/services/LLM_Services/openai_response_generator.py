"""
OpenAI Response Generator

This module provides the OpenAI implementation of the LLM response generator
using the Abstract Factory design pattern.
"""
from click import prompt
from openai import OpenAI, api_key
import os
import json
from typing import Dict, Any
from .llm_response_generator import LLMResponseGenerator


class OpenAIResponseGenerator(LLMResponseGenerator):
    """
    OpenAI implementation of the LLM response generator.
    
    This class provides concrete implementation for OpenAI's GPT models
    following the Abstract Factory design pattern.
    """
    
    def GenerateResponse(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response using OpenAI's GPT model.
        
        Args:
            prompt (str): The input prompt/question
            
        Returns:
            Dict[str, Any]: Response containing generated text and metadata
        """
        

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via 'api_key' argument or OPENAI_API_KEY environment variable.")

        

        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        client = OpenAI(api_key=api_key)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an academic assistant for university students. Always respond with valid JSON format containing 'answer' and 'status' fields."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500
            )
            raw_answer = resp.choices[0].message.content.strip()
            print(f"Raw LLM Response: {raw_answer}") 
            
            try:
                answer_json = json.loads(raw_answer)
                return {
                    "success": True,
                    "response": answer_json,
                    "model": model,
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens
                    }
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return structured response
                return {
                    "success": True,
                    "response": {
                        "answer": raw_answer,
                        "status": "complete"
                    },
                    "model": model,
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens
                    }
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": {
                    "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
                    "status": "error"
                }
            }
