import os
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
from app.core.logging import logger

def clean_exif_value(value):
    """
    Ensures EXIF values are JSON-serializable. Converts bytes and tuple types.
    """
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace').strip()
        except Exception:
            return str(value)
    elif isinstance(value, tuple):
        return [clean_exif_value(x) for x in value]
    elif isinstance(value, dict):
        return {str(k): clean_exif_value(v) for k, v in value.items()}
    return value

def extract_image_metadata(file_path: str) -> dict:
    """
    Extracts dimension, format, and EXIF tags from an image file.
    """
    metadata = {
        "dimensions": "Unknown",
        "format": "Unknown",
        "has_exif": False,
        "exif_details": {}
    }
    
    try:
        with Image.open(file_path) as img:
            metadata["dimensions"] = f"{img.width}x{img.height}"
            metadata["format"] = img.format or "Unknown"
            
            # Extract EXIF
            exif_data = img.getexif()
            if exif_data:
                metadata["has_exif"] = True
                raw_details = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    cleaned_val = clean_exif_value(value)
                    raw_details[tag_name] = cleaned_val
                metadata["exif_details"] = raw_details
                logger.info(f"Metadata: parsed {len(raw_details)} EXIF tags from image.")
    except Exception as e:
        logger.error(f"Failed to extract image metadata for {file_path}: {e}")
        
    return metadata

def extract_video_metadata(file_path: str) -> dict:
    """
    Extracts codec, resolution, fps, and duration from a video file.
    """
    metadata = {
        "resolution": "Unknown",
        "fps": 0.0,
        "duration_seconds": 0.0,
        "frame_count": 0,
        "codec": "Unknown"
    }
    
    try:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # FourCC codec conversion
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
            
            duration = frame_count / fps if fps > 0 else 0.0
            
            metadata["resolution"] = f"{width}x{height}"
            metadata["fps"] = round(fps, 2)
            metadata["frame_count"] = frame_count
            metadata["duration_seconds"] = round(duration, 2)
            metadata["codec"] = codec.strip() or "Unknown"
            
            cap.release()
            logger.info(f"Metadata: parsed video. Duration: {metadata['duration_seconds']}s, Codec: {metadata['codec']}")
    except Exception as e:
        logger.error(f"Failed to extract video metadata for {file_path}: {e}")
        
    return metadata

def extract_metadata(file_path: str, input_type: str) -> dict:
    """
    Orchestrates metadata extraction based on input type.
    """
    logger.info(f"Metadata Extractor: reading {file_path} as {input_type}")
    
    basic_meta = {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": 0
    }
    
    if os.path.exists(file_path):
        basic_meta["file_size_bytes"] = os.path.getsize(file_path)
        
    if input_type == "image":
        img_meta = extract_image_metadata(file_path)
        basic_meta.update(img_meta)
    elif input_type == "video":
        vid_meta = extract_video_metadata(file_path)
        basic_meta.update(vid_meta)
        
    return basic_meta
