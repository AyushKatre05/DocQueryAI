import asyncio
import logging
import re
from typing import List, Dict, Tuple
from io import BytesIO
import httpx
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=20.0)

    async def download_and_extract_text(self, pdf_url: str) -> str:
        try:
            response = await self.client.get(pdf_url)
            response.raise_for_status()
            if not response.content:
                raise ValueError("Empty PDF")
            return self._extract_text(response.content)
        except Exception as e:
            logger.error(str(e))
            raise ValueError(f"PDF processing failed: {str(e)}")

    def _extract_text(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for i in range(doc.page_count):
            t = self._clean(doc.load_page(i).get_text())
            if t:
                texts.append(t)
        doc.close()
        return "\n\n".join(texts)

    def _clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return "\n".join(l.strip() for l in text.split('\n') if len(l.strip()) > 10)

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) > chunk_size and cur:
                chunks.append(cur.strip())
                cur = cur[-overlap:] + " " + s
            else:
                cur = s if not cur else cur + " " + s
        if cur.strip():
            chunks.append(cur.strip())
        chunks = [c for c in chunks if len(c) > 50]
        return chunks or ([text.strip()] if text.strip() else [])


class EmbeddingManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.embeddings = None
        self.texts = []

    async def initialize(self):
        return

    async def create_index(self, texts: List[str]):
        self.texts = texts
        self.embeddings = self.vectorizer.fit_transform(texts)

    async def search_similar_contexts(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.embeddings).flatten()
        idx = np.argpartition(-sims, range(top_k))[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.texts[i], float(sims[i])) for i in idx]


class AIGenerator:
    def __init__(self):
        self.use_gemini = bool(settings.gemini_api_key)
        if self.use_gemini:
            try:
                from google import genai
                self.client = genai.Client(api_key=settings.gemini_api_key)
            except Exception:
                self.use_gemini = False

    async def generate_answer(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        if self.use_gemini:
            try:
                return await self._gemini(question, contexts)
            except Exception:
                pass
        return self._mock(question, contexts)

    async def _gemini(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        context_text = "\n\n".join(c[0] for c in contexts[:2])
        prompt = f"""Context:
{context_text}

Question: {question}

Return JSON with keys: answer, source_clause, explanation"""

        from google.genai import types
        import json

        r = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=400
            )
        )

        data = json.loads(r.text)
        return {
            "answer": data.get("answer", ""),
            "source_clause": data.get("source_clause", contexts[0][0][:200] if contexts else ""),
            "explanation": data.get("explanation", "")
        }

    def _mock(self, question: str, contexts: List[Tuple[str, float]]) -> Dict[str, str]:
        if not contexts:
            return {
                "answer": "No relevant information found.",
                "source_clause": "",
                "explanation": "No matching content."
            }
        ctx, score = contexts[0]
        src = ctx[:200].split('.')[0] + "."
        return {
            "answer": f"Based on the document, {question.lower().rstrip('?')} is described in the relevant section.",
            "source_clause": src,
            "explanation": f"Derived from document with relevance {score:.2f}."
        }
