import os
import shutil
from app.core.config import settings
from app.core.logging import logger

_reader = None
_has_tesseract = None

def check_tesseract() -> bool:
    """
    Checks if the Tesseract OCR binary is installed and present in the system PATH.
    """
    global _has_tesseract
    if _has_tesseract is None:
        _has_tesseract = shutil.which("tesseract") is not None
        if _has_tesseract:
            logger.info("Tesseract OCR binary detected in system PATH. Using pytesseract as primary OCR engine.")
        else:
            logger.warning("Tesseract binary not found in PATH. Pytesseract will be disabled.")
    return _has_tesseract

def get_ocr_reader():
    """
    Lazily loads the EasyOCR reader model to save RAM.
    """
    global _reader
    if _reader is None:
        if settings.DISABLE_EASYOCR:
            logger.info("EasyOCR Reader is disabled by configuration settings. Skipping initialization.")
            _reader = False
            return _reader
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
    Extracts text content from a local image file using Pytesseract (Primary) or EasyOCR (Secondary/Fallback).
    Returns empty string if both OCR engines fail or are unavailable.
    """
    # 1. Try Pytesseract if binary is available (Zero-RAM impact)
    if check_tesseract():
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(img)
            if extracted_text.strip():
                logger.info(f"Pytesseract OCR extracted text: '{extracted_text[:100]}...' ")
                return extracted_text.strip()
            else:
                logger.info("Pytesseract OCR finished, but found no text in image.")
                return ""
        except Exception as e:
            logger.error(f"Pytesseract extraction failed: {e}. Bypassing EasyOCR to prevent OOM crash.")
            return ""
            
    # 2. Fallback to EasyOCR (Heavy, PyTorch-based)
    reader = get_ocr_reader()
    if not reader:
        logger.warning("EasyOCR Reader is unavailable. Skipping text extraction.")
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
