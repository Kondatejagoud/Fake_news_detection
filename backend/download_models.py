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

    # 4. Vosk Speech-to-Text Model (Offline)
    try:
        if not os.path.exists(settings.VOSK_MODEL_PATH) or not os.listdir(settings.VOSK_MODEL_PATH):
            import urllib.request
            import zipfile
            zip_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            zip_path = "./cache/vosk-model.zip"
            print(f"Downloading Vosk model from {zip_url}...")
            urllib.request.urlretrieve(zip_url, zip_path)
            print("Extracting Vosk model zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("./cache/vosk/")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print("Vosk model downloaded and extracted successfully.")
        else:
            print("Vosk model already exists in cache.")
    except Exception as e:
        print(f"Warning: Failed to pre-download Vosk model: {e}")

    print("Pre-download process completed.")

if __name__ == "__main__":
    pre_download_models()
