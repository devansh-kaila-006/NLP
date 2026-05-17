"""
Gemini Generator - Generate LLM responses using Google Gemini
Creates grounded answers with citations
"""

import time
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    USE_NEW_API = True
except ImportError:
    import google.generativeai as genai
    from google.generativeai import types
    USE_NEW_API = False

from src.utils.logger import LoggerMixin
from src.utils.helpers import Timer
from src.config import LLM_CONFIG


class GeminiGenerator(LoggerMixin):
    """
    Generate responses using Google Gemini API
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Gemini generator

        Args:
            config: LLM configuration
        """
        # Load environment variables
        load_dotenv()

        self.config = config or LLM_CONFIG
        self.model_name = self.config.get("model", "gemini-2.0-flash")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.temperature = self.config.get("temperature", 0.3)
        self.timeout = self.config.get("timeout", 30)
        self.use_new_api = USE_NEW_API

        # Check API key
        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError("GOOGLE_API_KEY not set. Please set in .env file")

        # Strip 'models/' prefix if present
        if self.model_name.startswith("models/"):
            self.model_name = self.model_name.replace("models/", "")

        # Initialize Gemini
        if self.use_new_api:
            self.client = genai.Client(api_key=self.api_key)
            self.logger.info(f"Using new google.genai package with model: {self.model_name}")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.warning(f"Using deprecated google.generativeai package. Install google-genai for better performance.")

    def build_prompt(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        include_citations: bool = True
    ) -> str:
        """
        Build RAG prompt with context

        Args:
            question: User question
            chunks: Retrieved chunks
            include_citations: Include source citations

        Returns:
            Formatted prompt string
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get('source_name', 'Unknown')
            source_type = chunk.get('source_type', 'unknown')

            # Build citation based on source type
            if source_type == 'pdf':
                chapter = chunk.get('chapter', 'N/A')
                page = chunk.get('page_start', 'N/A')
                citation = f"[Source: {source}, Chapter {chapter}, Page {page}]"
            else:
                citation = f"[Source: {source}]"

            context_parts.append(f"{citation}\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a machine learning and deep learning teaching assistant.
Your task is to answer questions using the provided context from academic sources.

Context:
{context}

Question: {question}

Instructions:
- Answer the question clearly and concisely
- Use only the information from the context above
- Include citations to sources using the format shown in the context
- If the context doesn't contain enough information, say so
- Explain step-by-step for complex questions
- Include relevant technical details and formulas when present

Answer:"""

        return prompt

    def generate_answer(
        self,
        prompt: str,
        max_retries: int = 2
    ) -> str:
        """
        Generate answer from prompt

        Args:
            prompt: Input prompt
            max_retries: Number of retries on rate limit

        Returns:
            Generated answer text
        """
        self.logger.info(f"Generating answer (model: {self.model_name}, new_api: {self.use_new_api})")

        for attempt in range(max_retries):
            try:
                if self.use_new_api:
                    # Use new google.genai package (faster, more reliable)
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.config.get("max_output_tokens", 1024)
                        )
                    )

                    if response and hasattr(response, 'text') and response.text:
                        return response.text.strip()
                    else:
                        self.logger.warning("Empty response from Gemini")
                        return "I apologize, but I couldn't generate a response. Please try again."
                else:
                    # Use deprecated google.generativeai package
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.temperature,
                            candidate_count=1,
                            max_output_tokens=self.config.get("max_output_tokens", 1024)
                        )
                    )

                    if response and response.text:
                        return response.text.strip()
                    else:
                        self.logger.warning("Empty response from Gemini")
                        return "I apologize, but I couldn't generate a response. Please try again."

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting and server overload (503)
                if any(keyword in error_str for keyword in ["rate limit", "quota", "503", "unavailable", "overload"]):
                    if attempt < max_retries - 1:
                        wait_time = 3 + (attempt * 2)  # 3s, 5s (reduced from exponential)
                        self.logger.warning(f"Server overloaded/rate-limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error("Server unavailable after all retries")
                        return "I apologize, but the service is currently overloaded. Please try again in a minute."

                # Handle other errors
                self.logger.error(f"Error generating response: {e}")
                return f"I apologize, but I encountered an error: {str(e)}"

        return "Failed to generate response after multiple attempts."

    def format_response(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format final response with metadata

        Args:
            answer: Generated answer text
            chunks: Chunks used for generation

        Returns:
            Formatted response dictionary
        """
        # Extract source information
        sources = []
        seen_sources = set()

        for chunk in chunks:
            source_name = chunk.get('source_name', 'Unknown')

            if source_name not in seen_sources:
                seen_sources.add(source_name)

                source_info = {
                    "name": source_name,
                    "type": chunk.get('source_type', 'unknown'),
                    "relevance": chunk.get('rerank_score', chunk.get('relevance_score', 0))
                }

                # Add additional metadata
                if chunk.get('source_type') == 'pdf':
                    source_info['chapter'] = chunk.get('chapter', 'N/A')
                    source_info['page'] = chunk.get('page_start', 'N/A')
                elif chunk.get('source_type') in ['html', 'web']:
                    source_info['url'] = chunk.get('url', 'N/A')

                sources.append(source_info)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "model": self.model_name
        }

    def query(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end query: prompt + generate + format

        Args:
            question: User question
            chunks: Retrieved chunks
            include_citations: Include source citations

        Returns:
            Formatted response dictionary
        """
        self.logger.info(f"Processing query: '{question[:100]}...'")

        # Build prompt
        with Timer("Prompt building"):
            prompt = self.build_prompt(question, chunks, include_citations)

        # Generate answer
        with Timer("Answer generation"):
            answer = self.generate_answer(prompt)

        # Format response
        response = self.format_response(answer, chunks)

        self.logger.info(f"Query processed. Answer length: {len(answer)} chars")
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary with model info
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.config.get("max_output_tokens", 1024),
            "api_key_configured": bool(self.api_key and self.api_key != "your-api-key-here")
        }


if __name__ == "__main__":
    # Test Gemini generator
    try:
        generator = GeminiGenerator()

        # Test query
        question = "What is gradient descent?"
        test_chunks = [
            {
                "text": "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function.",
                "source_name": "DL_Textbook",
                "source_type": "pdf",
                "chapter": "4",
                "page_start": 78
            }
        ]

        response = generator.query(question, test_chunks)

        print(f"\nQuestion: {question}")
        print(f"\nAnswer:\n{response['answer']}")
        print(f"\nSources: {[s['name'] for s in response['sources']]}")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set GOOGLE_API_KEY in .env file")
