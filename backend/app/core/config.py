import os
from typing import List, Union, Optional
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hybrid Multimodal Fake Content Detection"
    API_PREFIX: str = "/api"
    GOOGLE_FACTCHECK_API_KEY: Optional[str] = None
    
    # Database
    DATABASE_URL: str = "sqlite:///./fake_detection.db"
    
    # CORS Origins
    # We allow string-separated lists or native lists
    ALLOWED_CORS_ORIGINS: Union[List[str], str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    @field_validator("ALLOWED_CORS_ORIGINS")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # Model caches and options
    HF_HOME: str = "./cache/huggingface"
    EASYOCR_CACHE: str = "./cache/easyocr"
    TORCH_HOME: str = "./cache/torch"
    VOSK_MODEL_PATH: str = "./cache/vosk/vosk-model-small-en-us-0.15"
    
    ENABLE_MOCK_FALLBACK: bool = True
    DISABLE_SENTENCE_TRANSFORMERS: bool = False
    DISABLE_EASYOCR: bool = False
    
    # Request limits (to prevent OOM on hobby containers)
    MAX_FILE_SIZE_MB: int = 15  # Limit video uploads to 15MB on free tier
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

# Ensure model cache dirs exist
os.makedirs(settings.HF_HOME, exist_ok=True)
os.makedirs(settings.EASYOCR_CACHE, exist_ok=True)
os.makedirs(settings.TORCH_HOME, exist_ok=True)
os.makedirs(os.path.dirname(settings.VOSK_MODEL_PATH), exist_ok=True)

# Set environment variables for huggingface and easyocr to use our caches
os.environ["HF_HOME"] = settings.HF_HOME
os.environ["TORCH_HOME"] = settings.TORCH_HOME
