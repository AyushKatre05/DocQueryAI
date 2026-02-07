import os
import json
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Intelligent Document Query System"
    debug: bool = False

    gemini_api_key: str = ""

    cors_origins: List[str] = ["*"]

    embedding_model: str = "intfloat/e5-small-v2"

    max_pdf_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    default_top_k: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        env = os.environ

        self.gemini_api_key = env.get("GEMINI_API_KEY", self.gemini_api_key)
        self.debug = env.get("DEBUG", str(self.debug)).lower() == "true"

        cors_env = env.get("CORS_ORIGINS")
        if cors_env:
            try:
                parsed = json.loads(cors_env)
                self.cors_origins = parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                self.cors_origins = [x.strip() for x in cors_env.split(",") if x.strip()]


settings = Settings()
