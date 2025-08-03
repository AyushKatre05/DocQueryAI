# config.py

import os
import json
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
    # Model settings
    embedding_model: str = "intfloat/e5-small-v2"


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

        # Load Gemini API key from environment
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)
        self.debug = os.getenv("DEBUG", str(self.debug)).lower() == "true"

        # Parse CORS origins (support JSON and CSV)
        cors_env = os.getenv("CORS_ORIGINS", json.dumps(self.cors_origins))
        try:
            parsed = json.loads(cors_env)
            self.cors_origins = parsed if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            self.cors_origins = [s.strip() for s in cors_env.split(",") if s.strip()]


# Instantiate globally
settings = Settings()
