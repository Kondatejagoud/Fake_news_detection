import os
import uuid
import tempfile
import cv2
from app.core.logging import logger

_mtcnn_detector = None

def get_face_detector():
    """
    Lazily loads MTCNN face detector.
    """
    global _mtcnn_detector
    if _mtcnn_detector is None:
        try:
            from mtcnn import MTCNN
            logger.info("Initializing MTCNN Face Detector...")
            _mtcnn_detector = MTCNN()
            logger.info("MTCNN Face Detector successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN detector: {e}. Falling back to OpenCV Haar Cascades.")
            _mtcnn_detector = False  # Sentinel for fallback
    return _mtcnn_detector

def detect_faces_in_frame(frame) -> list:
    """
    Detects faces in a single OpenCV frame.
    Uses MTCNN if available, otherwise falls back to OpenCV Haar Cascades.
    Returns a list of bounding boxes: [(x, y, w, h)]
    """
    detector = get_face_detector()
    
    if detector:
        try:
            # MTCNN expects RGB image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb_frame)
            boxes = []
            for d in detections:
                if d.get("confidence", 0) > 0.85:  # High confidence threshold
                    box = d["box"]  # [x, y, w, h]
                    boxes.append(tuple(box))
            return boxes
        except Exception as e:
            logger.warning(f"MTCNN face detection failed: {e}. Trying Haar Cascade fallback.")
            
    # Haar Cascade Fallback
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return [tuple(rect) for rect in detections]
    except Exception as e:
        logger.error(f"Haar Cascade face detection failed: {e}")
        return []

def extract_frames_and_faces(video_path: str, max_frames: int = 15) -> dict:
    """
    Samples frames from video (1 frame per second) and extracts face crops.
    Saves face crops to temp files and returns their paths.
    """
    logger.info(f"Video Frame Sampler: processing {video_path}")
    result = {
        "frames_extracted": 0,
        "faces_detected": 0,
        "face_image_paths": []
    }

    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return result

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0  # Fallback assumption
        
    sample_interval = int(fps)  # Sample 1 frame per second
    if sample_interval <= 0:
        sample_interval = 1

    temp_dir = tempfile.gettempdir()
    face_dir = os.path.join(temp_dir, "fake_detector_faces")
    os.makedirs(face_dir, exist_ok=True)

    frame_count = 0
    extracted_count = 0
    face_paths = []

    try:
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                extracted_count += 1
                
                # Detect faces on this sampled frame
                face_boxes = detect_faces_in_frame(frame)
                
                for (x, y, w, h) in face_boxes:
                    # Add padding to face crop for better CNN context
                    height, width, _ = frame.shape
                    pad_x = int(w * 0.15)
                    pad_y = int(h * 0.15)
                    
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(width, x + w + pad_x)
                    y2 = min(height, y + h + pad_y)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        face_filename = f"face_{uuid.uuid4()}.jpg"
                        face_path = os.path.join(face_dir, face_filename)
                        cv2.imwrite(face_path, face_crop)
                        face_paths.append(face_path)

            frame_count += 1
    finally:
        cap.release()
    
    result["frames_extracted"] = extracted_count
    result["faces_detected"] = len(face_paths)
    result["face_image_paths"] = face_paths
    
    logger.info(
        f"Video Frame Sampler finished: extracted {extracted_count} frames, "
        f"isolated {len(face_paths)} face crops."
    )
    return result
