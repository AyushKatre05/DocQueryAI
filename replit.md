# Overview

This is an intelligent document query system that processes PDF documents from URLs and answers questions using AI-powered semantic search. The system downloads PDFs, extracts text content, creates vector embeddings for semantic search, and generates contextual answers using OpenAI's GPT-4 with FAISS for efficient similarity search.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
- **FastAPI**: Modern Python web framework chosen for its async capabilities, automatic API documentation, and type safety
- **Async Architecture**: Built with async/await patterns for handling concurrent PDF downloads and processing
- **Lifespan Management**: Uses FastAPI's lifespan context manager to pre-load ML models during startup

## Document Processing Pipeline
- **PDF Download**: Uses httpx async client for downloading PDFs from URLs with 30-second timeout
- **Text Extraction**: PyMuPDF (fitz) for reliable PDF text extraction from downloaded bytes
- **Text Chunking**: Configurable chunk size (1000 chars) with overlap (200 chars) for better context preservation

## Vector Search System
- **Embeddings**: sentence-transformers with all-MiniLM-L6-v2 model (primary) or TF-IDF vectorizer (fallback) for creating semantic embeddings
- **Vector Database**: FAISS for efficient similarity search and retrieval
- **Search Strategy**: Top-k retrieval (default 3) for finding most relevant document chunks

## AI Answer Generation
- **Primary**: Google Gemini 2.5-flash for generating contextual answers from retrieved document chunks
- **Fallback**: Mock response system when Gemini API is unavailable
- **Context Integration**: Combines retrieved chunks with questions for comprehensive answers

## Configuration Management
- **Pydantic Settings**: Type-safe configuration with environment variable support
- **Environment-based**: Debug mode, API keys, and CORS settings configurable via environment
- **Validation**: Built-in validation for configuration parameters

## Error Handling
- **HTTP Exceptions**: Proper FastAPI exception handling with meaningful error messages
- **Logging**: Structured logging throughout the application for debugging and monitoring
- **Graceful Degradation**: Mock responses when external services are unavailable

# External Dependencies

## AI Services
- **Google Gemini API**: gemini-2.5-flash model for answer generation (with API key authentication)

## Machine Learning Libraries
- **sentence-transformers**: all-MiniLM-L6-v2 model for text embeddings
- **FAISS**: Facebook AI Similarity Search for vector operations
- **PyMuPDF (fitz)**: PDF text extraction library

## HTTP and Web
- **httpx**: Async HTTP client for PDF downloads
- **FastAPI**: Web framework with built-in CORS middleware
- **Uvicorn**: ASGI server for running the FastAPI application

## Data Processing
- **NumPy**: Numerical operations for vector processing
- **Pydantic**: Data validation and settings management

## Development and Deployment
- **Python 3.7+**: Runtime environment
- **Environment Variables**: For configuration management (.env file support)
- **Render**: Deployment platform (production-ready configuration)