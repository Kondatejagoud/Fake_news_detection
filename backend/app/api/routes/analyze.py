import os
import uuid
import tempfile
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db
from app.models.analysis import Analysis
from app.schemas.analysis import AnalysisResponse

# Import extraction layer
from app.extraction.url_scraper import scrape_url
from app.extraction.ocr import extract_text_from_image
from app.extraction.video_frames import extract_frames_and_faces
from app.extraction.metadata import extract_metadata
from app.extraction.audio_transcriber import extract_audio_to_wav, transcribe_audio_to_text

# Import analysis modules
from app.analysis.nlp_module import analyze_text
from app.analysis.image_module import analyze_image
from app.analysis.deepfake_module import analyze_deepfake

# Import fusion and explanation
from app.fusion.fusion_engine import fuse_results, compute_final_decision
from app.explanation.report_generator import generate_report

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    request: Request,
    db: Session = Depends(get_db)
):
    content_type = request.headers.get("content-type", "")
    
    input_type = None
    input_reference = ""
    file_path = None
    scraped_data = None
    
    # 1. PARSE REQUEST BY CONTENT-TYPE
    if "application/json" in content_type:
        try:
            body = await request.json()
            input_type = body.get("input_type")
            if input_type == "url":
                url_val = body.get("url")
                if not url_val:
                    raise HTTPException(status_code=400, detail="Missing 'url' field in JSON request.")
                input_reference = url_val
            elif input_type == "text":
                text_val = body.get("text")
                if not text_val:
                    raise HTTPException(status_code=400, detail="Missing 'text' field in JSON request.")
                input_reference = text_val[:100]  # Store first 100 chars as reference
                scraped_data = {"text": text_val, "title": "Raw text input", "images": [], "metadata": {}}
            else:
                raise HTTPException(status_code=400, detail=f"Invalid 'input_type': {input_type} for JSON body.")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON body: {str(e)}")
            
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()

            input_type = form.get("input_type")
            if input_type not in ["image", "video"]:
                raise HTTPException(status_code=400, detail="For file uploads, input_type must be 'image' or 'video'.")
                
            upload_file = form.get("file")
            from starlette.datastructures import UploadFile as StarletteUploadFile
            if not isinstance(upload_file, (UploadFile, StarletteUploadFile)):
                raise HTTPException(status_code=400, detail="Missing file parameter 'file' in multipart form.")
                
            # Basic file size validation
            # Read first chunk to check size
            chunk_size = 1024 * 1024  # 1MB
            size_bytes = 0
            
            # Temporary file write
            temp_dir = tempfile.gettempdir()
            filename = f"{uuid.uuid4()}_{upload_file.filename}"
            file_path = os.path.join(temp_dir, filename)
            input_reference = upload_file.filename or "uploaded_file"
            
            with open(file_path, "wb") as buffer:
                while chunk := await upload_file.read(chunk_size):
                    size_bytes += len(chunk)
                    if size_bytes > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                        # Clean up temp file
                        buffer.close()
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB."
                        )
                    buffer.write(chunk)
            
            logger.info(f"File uploaded successfully: {input_reference} ({size_bytes} bytes) saved to {file_path}")
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Error parsing multipart form: {str(e)}")
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type. Must be application/json or multipart/form-data."
        )

    # 2. RUN EXTRACTION AND ANALYSIS LAYERS
    modules_run = []
    module_results = {
        "text_nlp": None,
        "image_forensics": None,
        "deepfake": None
    }
    
    try:
        if input_type == "url":
            # Extract content from URL
            scraped_data = scrape_url(input_reference)
            text_to_analyze = scraped_data["text"]
            
            # Run text NLP
            nlp_res = analyze_text(text_to_analyze)
            module_results["text_nlp"] = nlp_res
            modules_run.append("text_nlp")
            
            # If the URL scraper successfully finds embedded images, analyze the first one
            if scraped_data.get("images"):
                logger.info(f"Scraped URL contains {len(scraped_data['images'])} images. Extracting and running forensics on the main image.")
                # Stub: we can run image forensics if downloading was implemented
                # For Phase 1 stubs, we stick to text.
                pass
                
        elif input_type == "text":
            # Analyze raw text
            text_to_analyze = scraped_data["text"]
            nlp_res = analyze_text(text_to_analyze)
            module_results["text_nlp"] = nlp_res
            modules_run.append("text_nlp")
            
        elif input_type == "image":
            # Extract metadata and OCR text
            metadata = extract_metadata(file_path, "image")
            ocr_text = extract_text_from_image(file_path)
            
            # Run Image Forensics
            img_res = analyze_image(file_path)
            module_results["image_forensics"] = img_res
            modules_run.append("image_forensics")
            
            # If text is found via OCR, run NLP analysis as well
            if ocr_text.strip():
                logger.info(f"OCR found text in image: '{ocr_text[:50]}...'. Running NLP analysis.")
                nlp_res = analyze_text(ocr_text)
                module_results["text_nlp"] = nlp_res
                modules_run.append("text_nlp")
        
        elif input_type == "video":
            # Extract frames and crop faces
            frame_data = extract_frames_and_faces(file_path)
            faces_paths = frame_data.get("face_image_paths", [])
            
            # Run Deepfake classifier
            df_res = analyze_deepfake(faces_paths)
            module_results["deepfake"] = df_res
            modules_run.append("deepfake")
            
            # Extract and transcribe spoken content (ASR)
            temp_dir = tempfile.gettempdir()
            temp_wav_path = os.path.join(temp_dir, f"{uuid.uuid4()}_temp_audio.wav")
            try:
                if extract_audio_to_wav(file_path, temp_wav_path):
                    spoken_text = transcribe_audio_to_text(temp_wav_path)
                    if spoken_text.strip():
                        logger.info(f"Video contains spoken content: '{spoken_text[:100]}...'. Running NLP analysis.")
                        nlp_res = analyze_text(spoken_text)
                        module_results["text_nlp"] = nlp_res
                        modules_run.append("text_nlp")
                    else:
                        logger.info("Video contains no detectable spoken content.")
            finally:
                if os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except Exception as ex:
                        logger.warning(f"Failed to delete temp wav file {temp_wav_path}: {ex}")
            
    except Exception as e:
        logger.exception(f"Error during content extraction or analysis: {e}")
        # Clean up temp file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Content extraction/analysis pipeline failure: {str(e)}"
        )
        
    # Clean up upload file now that analysis is done
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary upload file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {file_path}: {e}")

    # 3. CENTRALIZED FINAL DECISION ENGINE
    reduce_confidence = False
    if module_results["text_nlp"]:
        reduce_confidence = module_results["text_nlp"].get("reduce_confidence", False)
        
    decision = compute_final_decision(
        input_type=input_type,
        module_results=module_results,
        reduce_confidence=reduce_confidence
    )
    
    authenticity_score = decision["authenticity_score"]
    confidence_percentage = decision["confidence_percentage"]
    risk_level = decision["risk_level"]
    verdict = decision["verdict"]
    explanation_list = decision["explanation"]
    supporting_sources = decision["supporting_sources"]
    
    # Store final decision inside module_results JSON column for database persistence
    module_results["final_decision"] = {
        "verdict": verdict,
        "supporting_sources": supporting_sources
    }

    # 4. DB SAVE
    try:
        new_analysis = Analysis(
            input_type=input_type,
            input_reference=input_reference,
            authenticity_score=authenticity_score,
            confidence_percentage=confidence_percentage,
            risk_level=risk_level,
            module_results=module_results,
            explanation=explanation_list
        )
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)
        logger.info(f"Successfully saved analysis record to db: {new_analysis.id}")
        return new_analysis.to_dict()
    except Exception as e:
        logger.error(f"Failed to commit analysis database record: {e}")
        # Return transient response if database commit fails (graceful operation)
        return {
            "analysis_id": str(uuid.uuid4()),
            "input_type": input_type,
            "input_reference": input_reference,
            "authenticity_score": authenticity_score,
            "confidence_percentage": confidence_percentage,
            "risk_level": risk_level,
            "modules_run": modules_run,
            "module_results": module_results,
            "explanation": explanation_list,
            "verdict": verdict,
            "supporting_sources": supporting_sources,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

@router.get("/analyses", response_model=List[AnalysisResponse])
def get_analyses_history(db: Session = Depends(get_db)):
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(20).all()
    return [a.to_dict() for a in analyses]

@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
def get_analysis_by_id(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found."
        )
    return analysis.to_dict()
