import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    app_name: str = "Intelligent Document Query System"
    debug: bool = False
    
    # API Keys
    gemini_api_key: str = ""
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # PDF processing settings
    max_pdf_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Search settings
    default_top_k: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get Gemini API key from environment with fallback
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Set debug mode based on environment
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Parse CORS origins from environment
        cors_env = os.getenv("CORS_ORIGINS", "*")
        if cors_env != "*":
            self.cors_origins = [origin.strip() for origin in cors_env.split(",")]

# Global settings instance
settings = Settings()
