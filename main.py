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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
pdf_processor = None
embedding_manager = None
ai_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global pdf_processor, embedding_manager, ai_generator
    
    logger.info("Initializing application components...")
    pdf_processor = PDFProcessor()
    embedding_manager = EmbeddingManager()
    ai_generator = AIGenerator()
    await embedding_manager.initialize()
    
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")

# FastAPI app
app = FastAPI(
    title="Intelligent Document Query System",
    description="FastAPI-powered intelligent document query system with vector embeddings and AI-powered answers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response Models
class QuestionAnswerRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer")

@app.get("/")
async def root():
    return {
        "message": "Intelligent Document Query System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "pdf_processor": pdf_processor is not None,
            "embedding_manager": embedding_manager is not None and embedding_manager.model is not None,
            "ai_generator": ai_generator is not None
        }
    }

@app.post("/api/v1/hackrx/run")
async def process_document_questions(request: QuestionAnswerRequest):
    """
    HackRx Challenge Endpoint:
    Given a PDF and list of questions, return only plain answers in the correct format.
    """
    try:
        logger.info(f"Processing document: {request.documents}")
        logger.info(f"Questions count: {len(request.questions)}")
        
        # Step 1: Download and extract text from PDF
        pdf_text = await pdf_processor.download_and_extract_text(str(request.documents))
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Step 2: Chunk the document
        chunks = pdf_processor.chunk_text(pdf_text)
        logger.info(f"Document chunked into {len(chunks)} segments")
        
        # Step 3: Create FAISS index
        await embedding_manager.create_index(chunks)

        # Step 4: Answer each question (return only `answer` field)
        final_answers = []
        for question in request.questions:
            try:
                logger.info(f"Processing question: {question}")
                relevant_contexts = await embedding_manager.search_similar_contexts(question, top_k=3)

                if not relevant_contexts:
                    final_answers.append("I couldn't find relevant information in the document to answer this question.")
                    continue

                ai_response = await ai_generator.generate_answer(question, relevant_contexts)
                final_answers.append(ai_response["answer"])
            except Exception as e:
                logger.error(f"Error answering question: {question} | Error: {str(e)}")
                final_answers.append("An error occurred while processing this question.")
        
        logger.info("Finished processing all questions")
        return {"answers": final_answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc) if settings.debug else "An unexpected error occurred"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level="info"
    )
