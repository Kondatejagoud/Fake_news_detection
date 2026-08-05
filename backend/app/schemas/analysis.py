from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any
from datetime import datetime

class AnalysisRequest(BaseModel):
    input_type: str = Field(..., description="Type of input: 'url' or 'text'")
    url: Optional[str] = Field(None, description="URL to analyze, required if input_type is 'url'")
    text: Optional[str] = Field(None, description="Raw text to analyze, required if input_type is 'text'")

class ModuleTextNLPResult(BaseModel):
    score: float = Field(..., description="Fake probability score between 0.0 and 1.0")
    flagged_claims: List[str] = Field(default_factory=list, description="Claims flagging misinformation patterns")

class ModuleImageForensicsResult(BaseModel):
    score: float = Field(..., description="Fake probability score between 0.0 and 1.0")
    manipulated_regions: int = Field(0, description="Count of suspected manipulated regions detected")

class ModuleDeepfakeResult(BaseModel):
    score: float = Field(..., description="Fake probability score between 0.0 and 1.0")
    faces_detected: int = Field(0, description="Number of faces scanned for deepfakes")

class ModuleResults(BaseModel):
    text_nlp: Optional[ModuleTextNLPResult] = None
    image_forensics: Optional[ModuleImageForensicsResult] = None
    deepfake: Optional[ModuleDeepfakeResult] = None

class AnalysisResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique UUID for this analysis task")
    input_type: str = Field(..., description="Input type: 'url', 'text', 'image', or 'video'")
    input_reference: str = Field(..., description="URL parsed or name of file uploaded")
    authenticity_score: int = Field(..., description="Authenticity rating between 0 and 100 (high is authentic)")
    confidence_percentage: int = Field(..., description="Confidence rating between 0 and 100")
    risk_level: str = Field(..., description="Risk class: Low, Medium, High")
    modules_run: List[str] = Field(..., description="Identifiers of modules that processed this request")
    module_results: Dict[str, Any] = Field(..., description="Individual module score outputs")
    processing_time: float = Field(0.0, description="Real request execution duration in seconds")
    
    # Centralized Final Decision fields
    verdict: str = Field(..., description="Unified authenticity verdict (TRUE, FALSE, NEEDS REVIEW, UNVERIFIED)")
    recommendation: str = Field(..., description="Dynamic recommendation based on verdict")
    explanation: List[str] = Field(..., description="Plain-language bullet points justifying the authenticity score")
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence publisher items")
    contradicting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Contradicting evidence publisher items")
    confidence: int = Field(..., description="Dynamic confidence rating matching actual consensus index")
    evidence_sources: List[str] = Field(default_factory=list, description="List of source publishers that provided data")
    primary_entity: str = Field(..., description="Primary entity identified")
    named_entities: List[str] = Field(default_factory=list, description="Formatted list of entities (Name (TYPE))")
    
    # Backward compatibility fields
    supporting_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Citations of supporting evidence sources")
    created_at: str = Field(..., description="Timestamp of completion")
