from app.core.config import settings
from app.core.logging import logger

_reader = None

def get_ocr_reader():
    """
    Lazily loads the EasyOCR reader model to save RAM.
    """
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR Reader (English, CPU mode)...")
            _reader = easyocr.Reader(
                ['en'], 
                gpu=False, 
                model_storage_directory=settings.EASYOCR_CACHE, 
                download_enabled=True
            )
            logger.info("EasyOCR Reader successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR Reader: {e}. OCR features will be disabled.")
            _reader = False  # Sentinel for failed load
    return _reader

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text content from a local image file using EasyOCR.
    Returns empty string if OCR fails or is unavailable.
    """
    reader = get_ocr_reader()
    if not reader:
        logger.warning("OCR Reader is unavailable. Skipping text extraction.")
        return ""
        
    try:
        # readtext returns a list of tuples: (bbox, text, confidence)
        results = reader.readtext(image_path)
        extracted_text = " ".join([res[1] for res in results])
        logger.info(f"EasyOCR extracted {len(results)} text segments. Combined: '{extracted_text[:100]}...'")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"EasyOCR text extraction failed for {image_path}: {e}")
        return ""
