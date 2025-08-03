import asyncio
import logging
import re
from typing import List, Dict, Any, Tuple
import tempfile
import os
from io import BytesIO
import httpx
import fitz  # PyMuPDF
import numpy as np
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def download_and_extract_text(self, pdf_url: str) -> str:
        try:
            logger.info(f"Downloading PDF from: {pdf_url}")
            response = await self.client.get(pdf_url)
            response.raise_for_status()

            if not response.content:
                raise ValueError("Downloaded PDF is empty")

            text = self._extract_text_from_bytes(response.content)
            logger.info(f"Extracted {len(text)} characters from PDF")

            return text

        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF: {str(e)}")
            raise ValueError(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")

    def _extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = self._clean_text(page.get_text())
                if text.strip():
                    text_parts.append(text)
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        return '\n'.join(cleaned_lines)

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        filtered_chunks = [chunk for chunk in chunks if len(chunk) > 50]

        if not filtered_chunks and chunks:
            filtered_chunks = [max(chunks, key=len)]

        if not filtered_chunks and text.strip():
            filtered_chunks = [text.strip()]

        logger.info(f"Created {len(filtered_chunks)} text chunks")
        return filtered_chunks


class EmbeddingManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.embeddings = None
        self.texts = []

    async def initialize(self):
        pass  # No model to load

    async def create_index(self, texts: List[str]):
        self.texts = texts
        self.embeddings = self.vectorizer.fit_transform(texts)

    async def search_similar_contexts(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.texts[i], similarities[i]) for i in top_indices]


class AIGenerator:
    def __init__(self):
        self.use_gemini = bool(settings.gemini_api_key)
        if self.use_gemini:
            try:
                from google import genai
                self.client = genai.Client(api_key=settings.gemini_api_key)
                logger.info("Gemini client initialized")
            except ImportError:
                logger.warning("Google genai package not available, using mock responses")
                self.use_gemini = False
        else:
            logger.info("Gemini API key not provided, using mock responses")

    async def generate_answer(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        try:
            if self.use_gemini:
                return await self._generate_gemini_answer(question, contexts)
            else:
                return self._generate_mock_answer(question, contexts)
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return {
                "answer": "I encountered an error while generating the answer.",
                "source_clause": "Error in processing",
                "explanation": f"Technical error occurred: {str(e)}"
            }

    async def _generate_gemini_answer(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        try:
            context_text = "\n\n".join([ctx[0] for ctx in contexts[:2]])
            prompt = f"""Based on the following document context, answer the question accurately and concisely.

Context:
{context_text}

Question: {question}

Please provide a JSON response with the following structure:
{{
    "answer": "Your detailed answer based on the context",
    "source_clause": "The specific clause or section from the document that supports your answer",
    "explanation": "Brief explanation of how you derived the answer from the document"
}}

If the context doesn't contain enough information to answer the question, say so clearly."""

            from google.genai import types

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert document analyzer.",
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=500
                )
            )

            import json
            result = json.loads(response.text)

            return {
                "answer": result.get("answer", "Unable to generate answer"),
                "source_clause": result.get("source_clause", contexts[0][0][:200] + "..." if contexts else "No source available"),
                "explanation": result.get("explanation", "Answer generated using AI analysis")
            }

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return self._generate_mock_answer(question, contexts)

    def _generate_mock_answer(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        if not contexts:
            return {
                "answer": "I couldn't find relevant information in the document to answer this question.",
                "source_clause": "No relevant context found",
                "explanation": "The question didn't match any content in the provided document."
            }

        best_context, score = contexts[0]
        answer_templates = [
            f"Based on the document content, {question.lower().replace('?', '')} can be found in the relevant section.",
            f"According to the document, the information regarding {question.lower().replace('?', '')} is detailed in the source material.",
            f"The document provides information about {question.lower().replace('?', '')} in the referenced section."
        ]

        import random
        answer = random.choice(answer_templates)

        source_clause = best_context[:200] + "..." if len(best_context) > 200 else best_context
        if '.' in source_clause:
            source_clause = source_clause.split('.')[0] + "."

        return {
            "answer": answer,
            "source_clause": source_clause,
            "explanation": f"Answer derived from document analysis with {score:.2f} relevance score. Note: This is a simulated response as Gemini API key is not configured."
        }
