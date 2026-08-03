import os
import sys

# Ensure backend folder is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

def pre_download_models():
    print("Pre-downloading machine learning model weights to cache directories...")
    
    # 1. Download HuggingFace Transformers (DistilBERT)
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        print(f"Downloading transformers model '{model_name}' to {settings.HF_HOME}...")
        AutoTokenizer.from_pretrained(model_name, cache_dir=settings.HF_HOME)
        AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=settings.HF_HOME)
        print("Transformers model downloaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to pre-download Transformers model: {e}")

    # 2. Download EasyOCR Models
    try:
        import easyocr
        print(f"Downloading EasyOCR models (english) to {settings.EASYOCR_CACHE}...")
        # Initializing easyocr.Reader automatically downloads detection and recognition models
        easyocr.Reader(['en'], model_storage_directory=settings.EASYOCR_CACHE, download_enabled=True)
        print("EasyOCR models downloaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to pre-download EasyOCR models: {e}")

    # 3. Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        smodel_name = "all-MiniLM-L6-v2"
        print(f"Downloading sentence-transformers model '{smodel_name}' to {settings.HF_HOME}...")
        SentenceTransformer(smodel_name, cache_dir=settings.HF_HOME)
        print("Sentence Transformers model downloaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to pre-download Sentence Transformers model: {e}")

    print("Pre-download process completed.")

if __name__ == "__main__":
    pre_download_models()
