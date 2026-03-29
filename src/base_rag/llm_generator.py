"""
Module: base_rag.llm_generator

Description:
    LLM integration for response generation using retrieved context

Inputs:
    - Query
    - Retrieved chunks

Outputs:
    - Generated response with citations

Dependencies:
    - google.generativeai
    - typing
    - src.utils.logger
    - src.utils.exceptions

Usage:
    >>> from src.base_rag.llm_generator import LLMGenerator
    >>> generator = LLMGenerator()
    >>> response = generator.generate(query, retrieved_chunks)
"""

import os
from typing import Dict, List

import google.generativeai as genai

from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """
    Generate responses using Google Gemini API with retrieved context.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize LLM generator.

        Args:
            model_name: Name of Gemini model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Raises:
            LLMError: If API key not found or model fails to initialize
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError(
                "GOOGLE_API_KEY environment variable not set",
                details={"required_var": "GOOGLE_API_KEY"}
            )

        # Configure API
        genai.configure(api_key=api_key)

        # Initialize model
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized LLM: {model_name}")
        except Exception as e:
            raise LLMError(
                f"Failed to initialize LLM: {str(e)}",
                details={"model_name": model_name, "error": str(e)}
            )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]],
        include_citations: bool = True
    ) -> Dict[str, any]:
        """
        Generate response for query using retrieved chunks as context.

        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks with metadata
            include_citations: Whether to include citations in response

        Returns:
            Dictionary with response text and metadata

        Raises:
            LLMError: If generation fails

        Example:
            >>> generator = LLMGenerator()
            >>> response = generator.generate(
            ...     "What is a CNN?",
            ...     retrieved_chunks
            ... )
            >>> print(response["text"])
        """
        try:
            logger.info(f"Generating response for query: {query}")

            # Construct prompt
            prompt = self._construct_prompt(query, retrieved_chunks, include_citations)

            # Generate response
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )

            result = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            response_text = result.text

            # Extract citations if enabled
            citations = []
            if include_citations:
                citations = self._extract_citations(retrieved_chunks)

            logger.info("Response generated successfully")

            return {
                "text": response_text,
                "query": query,
                "sources_used": len(retrieved_chunks),
                "citations": citations,
                "model": self.model_name
            }

        except Exception as e:
            raise LLMError(
                f"Generation failed: {str(e)}",
                details={"query": query, "error": str(e)}
            )

    def _construct_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]],
        include_citations: bool
    ) -> str:
        """
        Construct prompt with query and retrieved context.

        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
            include_citations: Whether to include citation instructions

        Returns:
            Constructed prompt string
        """
        # System prompt
        system_prompt = self._get_system_prompt()

        # Context from retrieved chunks
        context = self._format_context(retrieved_chunks, include_citations)

        # Citation instructions
        citation_instructions = self._get_citation_instructions() if include_citations else ""

        # Construct full prompt
        prompt = f"""{system_prompt}

Context from relevant documents:
{context}

{citation_instructions}

Question: {query}

Answer:"""

        return prompt

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for the LLM.

        Returns:
            System prompt string
        """
        return """You are a helpful teaching assistant for machine learning and deep learning topics. Your role is to provide clear, accurate, and educational answers to student questions.

Guidelines:
- Be thorough but concise
- Use simple language when possible, but don't oversimplify complex concepts
- Provide examples when helpful
- If the context doesn't contain enough information, say so
- Always base your answers on the provided context
- Explain technical terms when you first use them"""

    def _format_context(
        self,
        chunks: List[Dict[str, any]],
        include_citations: bool
    ) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: Retrieved chunks
            include_citations: Whether to include citation markers

        Returns:
            Formatted context string
        """
        formatted_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            # Get chunk text
            text = chunk.get("text", "")

            # Get metadata
            source = chunk.get("source", "Unknown")
            chunk_type = chunk.get("chunk_type", "unknown")

            # Format citation marker
            if include_citations:
                if chunk_type == "pdf":
                    citation = f"[PDF{i}: {source}, page {chunk.get('page', 'N/A')}]"
                else:  # video
                    timestamp = chunk.get("start_time", "N/A")
                    citation = f"[Video{i}: {source}, {timestamp}]"
            else:
                citation = f"[Source {i}]"

            # Format chunk
            formatted_chunk = f"{citation}\n{text}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n".join(formatted_chunks)

    def _get_citation_instructions(self) -> str:
        """
        Get instructions for citing sources.

        Returns:
            Citation instructions string
        """
        return """When answering the question:
- Use the information from the provided context
- Cite your sources using the citation markers in square brackets
- For example: "CNNs are used for image processing [PDF1: cs231n_lecture1, page 5]"
- If multiple sources support the same point, cite all of them
- Do not make up information that isn't in the context"""

    def _extract_citations(self, chunks: List[Dict[str, any]]) -> List[Dict[str, str]]:
        """
        Extract citation information from chunks.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of citation dictionaries
        """
        citations = []

        for i, chunk in enumerate(chunks, start=1):
            citation = {
                "id": i,
                "source": chunk.get("source", "Unknown"),
                "type": chunk.get("chunk_type", "unknown"),
                "page": chunk.get("page", None),
                "timestamp": chunk.get("start_time", None),
                "score": chunk.get("score", 0)
            }
            citations.append(citation)

        return citations

    def generate_with_sources(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]]
    ) -> str:
        """
        Generate response and format with source list.

        Args:
            query: User query
            retrieved_chunks: Retrieved chunks

        Returns:
            Formatted response with source list

        Example:
            >>> generator = LLMGenerator()
            >>> response = generator.generate_with_sources(
            ...     "What is backpropagation?",
            ...     chunks
            ... )
            >>> print(response)
        """
        # Generate response
        result = self.generate(query, retrieved_chunks, include_citations=True)

        # Format response with sources
        response_text = result["text"]
        citations = result["citations"]

        # Add source list
        if citations:
            source_list = "\n\nSources:\n"
            for citation in citations:
                if citation["type"] == "pdf":
                    source_list += f"- {citation['type'].upper()}: {citation['source']}, page {citation['page']}\n"
                else:  # video
                    source_list += f"- {citation['type'].upper()}: {citation['source']}, timestamp {citation['timestamp']}\n"

            response_text += source_list

        return response_text

    def batch_generate(
        self,
        queries: List[str],
        all_retrieved_chunks: List[List[Dict[str, any]]]
    ) -> List[Dict[str, any]]:
        """
        Generate responses for multiple queries.

        Args:
            queries: List of queries
            all_retrieved_chunks: List of retrieved chunks for each query

        Returns:
            List of response dictionaries

        Example:
            >>> generator = LLMGenerator()
            >>> queries = ["What is a CNN?", "Explain RNNs"]
            >>> chunks_list = [chunks1, chunks2]
            >>> responses = generator.batch_generate(queries, chunks_list)
        """
        responses = []

        for query, chunks in zip(queries, all_retrieved_chunks):
            try:
                response = self.generate(query, chunks)
                responses.append(response)
            except LLMError as e:
                logger.error(f"Failed to generate for query '{query}': {e}")
                responses.append({
                    "text": f"Error generating response: {str(e)}",
                    "query": query,
                    "error": str(e)
                })

        return responses
