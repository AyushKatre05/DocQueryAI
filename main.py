import asyncio
import logging
from typing import List
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import uvicorn

from utils.utils import PDFProcessor, EmbeddingManager, AIGenerator
from config.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

pdf_processor = None
embedding_manager = None
ai_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pdf_processor, embedding_manager, ai_generator
    pdf_processor = PDFProcessor()
    embedding_manager = EmbeddingManager()
    ai_generator = AIGenerator()
    await embedding_manager.initialize()
    yield


app = FastAPI(
    title="Intelligent Document Query System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionAnswerRequest(BaseModel):
    documents: HttpUrl = Field(...)
    questions: List[str] = Field(..., min_items=1)


@app.get("/")
async def root():
    return {"message": "running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pdf": pdf_processor is not None,
        "embedding": embedding_manager is not None,
        "ai": ai_generator is not None
    }


@app.post("/api/v1/hackrx/run")
async def process(request: QuestionAnswerRequest):
    try:
        pdf_text = await pdf_processor.download_and_extract_text(str(request.documents))
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="Empty PDF text")

        chunks = pdf_processor.chunk_text(pdf_text)
        await embedding_manager.create_index(chunks)

        async def answer_one(q: str):
            try:
                ctx = await embedding_manager.search_similar_contexts(q, top_k=settings.default_top_k)
                if not ctx:
                    return "I couldn't find relevant information in the document."
                r = await ai_generator.generate_answer(q, ctx)
                return r.get("answer", "")
            except Exception as e:
                logger.error(str(e))
                return "Error processing question."

        answers = await asyncio.gather(*(answer_one(q) for q in request.questions))
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(str(exc))
    return {"error": "Internal server error"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=settings.debug)
