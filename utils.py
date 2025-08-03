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
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Simple text vectorizer fallback
    from sklearn.feature_extraction.text import TfidfVectorizer

from config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF downloading and text extraction"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def download_and_extract_text(self, pdf_url: str) -> str:
        """Download PDF from URL and extract text"""
        try:
            logger.info(f"Downloading PDF from: {pdf_url}")
            
            # Download PDF
            response = await self.client.get(pdf_url)
            response.raise_for_status()
            
            if not response.content:
                raise ValueError("Downloaded PDF is empty")
            
            # Extract text using PyMuPDF
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
        """Extract text from PDF bytes using PyMuPDF"""
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean up text
                text = self._clean_text(text)
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic cleanup)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) > 10:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        if not text:
            return []
        
        # Split by sentences first to preserve context
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start a new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap by keeping some sentences from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks, but keep at least one chunk if text exists
        filtered_chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        # If no chunks meet the length requirement but we have text, keep the longest chunk
        if not filtered_chunks and chunks:
            filtered_chunks = [max(chunks, key=len)]
        
        # If still no chunks but we have text, create one chunk from all text
        if not filtered_chunks and text.strip():
            filtered_chunks = [text.strip()]
        
        logger.info(f"Created {len(filtered_chunks)} text chunks")
        return filtered_chunks

class EmbeddingManager:
    """Manages sentence embeddings and FAISS vector store"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.use_transformers = HAS_SENTENCE_TRANSFORMERS
        self.model_name = "all-MiniLM-L6-v2"  # Best balance of speed/performance in 2025
    
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            if self.use_transformers:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                
                # Load model in a thread to avoid blocking
                def load_model():
                    return SentenceTransformer(self.model_name)
                
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(None, load_model)
                
                logger.info("Sentence transformer model loaded successfully")
            else:
                logger.info("Using TF-IDF vectorizer as fallback")
                self.model = TfidfVectorizer(
                    max_features=384, 
                    stop_words='english',
                    min_df=1,
                    ngram_range=(1, 2)
                )
                logger.info("TF-IDF vectorizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise ValueError(f"Failed to initialize embedding model: {str(e)}")
    
    async def create_index(self, chunks: List[str]):
        """Create FAISS index from text chunks"""
        try:
            if not self.model:
                await self.initialize()
            
            self.chunks = chunks
            logger.info(f"Creating embeddings for {len(chunks)} chunks")
            
            # Generate embeddings in a thread to avoid blocking
            def generate_embeddings():
                if self.use_transformers:
                    embeddings = self.model.encode(chunks, show_progress_bar=False)
                    return embeddings.astype('float32')
                else:
                    # Use TF-IDF as fallback
                    try:
                        tfidf_matrix = self.model.fit_transform(chunks)
                        if tfidf_matrix.shape[1] == 0:
                            # Fallback: create simple word count vectors
                            logger.warning("TF-IDF failed, using simple word count vectors")
                            vocab = set()
                            for chunk in chunks:
                                vocab.update(chunk.lower().split())
                            vocab = list(vocab)[:384]  # Limit vocabulary size
                            self._simple_vocab = vocab  # Store for query processing
                            
                            vectors = []
                            for chunk in chunks:
                                words = chunk.lower().split()
                                vector = [words.count(word) for word in vocab]
                                vectors.append(vector)
                            
                            return np.array(vectors, dtype='float32')
                        
                        return np.array(tfidf_matrix.todense(), dtype='float32')
                    except Exception as e:
                        logger.error(f"TF-IDF failed: {e}, using simple word count vectors")
                        # Simple fallback: word count vectors
                        vocab = set()
                        for chunk in chunks:
                            vocab.update(chunk.lower().split())
                        vocab = list(vocab)[:384]  # Limit vocabulary size
                        self._simple_vocab = vocab  # Store for query processing
                        
                        vectors = []
                        for chunk in chunks:
                            words = chunk.lower().split()
                            vector = [words.count(word) for word in vocab]
                            vectors.append(vector)
                        
                        return np.array(vectors, dtype='float32')
            
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, generate_embeddings)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            self.index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            raise ValueError(f"Failed to create embeddings index: {str(e)}")
    
    async def search_similar_contexts(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar contexts using vector similarity"""
        try:
            if not self.model or not self.index:
                raise ValueError("Embedding model or index not initialized")
            
            # Generate query embedding
            def generate_query_embedding():
                if self.use_transformers:
                    query_vector = self.model.encode([query])
                    return query_vector.astype('float32')
                else:
                    # Use TF-IDF as fallback
                    try:
                        query_vector = self.model.transform([query])
                        if query_vector.shape[1] == 0:
                            raise ValueError("No vocabulary available")
                        return np.array(query_vector.todense(), dtype='float32')
                    except Exception as e:
                        logger.warning(f"Query embedding failed: {e}, using simple approach")
                        # Simple word matching approach
                        query_words = query.lower().split()
                        # Get vocabulary from the fitted model or create simple vector
                        if hasattr(self, '_simple_vocab'):
                            vocab = self._simple_vocab
                        else:
                            vocab = query_words  # Use query words as vocab
                        
                        vector = [query_words.count(word) for word in vocab]
                        # Pad or truncate to match expected dimensions
                        while len(vector) < self.index.d:
                            vector.append(0.0)
                        vector = vector[:self.index.d]
                        
                        return np.array([vector], dtype='float32')
            
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(None, generate_query_embedding)
            
            # Normalize query vector
            faiss.normalize_L2(query_vector)
            
            # Search similar contexts
            scores, indices = self.index.search(query_vector, top_k)
            
            # Return contexts with scores
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    context = self.chunks[idx]
                    score = float(scores[0][i])
                    results.append((context, score))
            
            logger.info(f"Found {len(results)} similar contexts for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar contexts: {str(e)}")
            return []

class AIGenerator:
    """Generates AI-powered answers using Gemini or mock responses"""
    
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
        """Generate answer based on question and relevant contexts"""
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
        """Generate answer using Gemini API"""
        try:
            # Prepare context from top results
            context_text = "\n\n".join([ctx[0] for ctx in contexts[:2]])  # Use top 2 contexts
            
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

            # Note that the newest Gemini model series is "gemini-2.5-flash" or "gemini-2.5-pro"
            # do not change this unless explicitly requested by the user
            from google.genai import types
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert document analyzer. Provide accurate, well-sourced answers based on the given context.",
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
            # Fallback to mock response
            return self._generate_mock_answer(question, contexts)
    
    def _generate_mock_answer(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        """Generate mock answer when OpenAI is not available"""
        if not contexts:
            return {
                "answer": "I couldn't find relevant information in the document to answer this question.",
                "source_clause": "No relevant context found",
                "explanation": "The question didn't match any content in the provided document."
            }
        
        # Use the best matching context
        best_context, score = contexts[0]
        
        # Generate a template-based answer
        answer_templates = [
            f"Based on the document content, {question.lower().replace('?', '')} can be found in the relevant section.",
            f"According to the document, the information regarding {question.lower().replace('?', '')} is detailed in the source material.",
            f"The document provides information about {question.lower().replace('?', '')} in the referenced section."
        ]
        
        import random
        answer = random.choice(answer_templates)
        
        # Extract a meaningful source clause (first sentence or up to 200 chars)
        source_clause = best_context[:200] + "..." if len(best_context) > 200 else best_context
        if '.' in source_clause:
            source_clause = source_clause.split('.')[0] + "."
        
        return {
            "answer": answer,
            "source_clause": source_clause,
            "explanation": f"Answer derived from document analysis with {score:.2f} relevance score. Note: This is a simulated response as Gemini API key is not configured."
        }
