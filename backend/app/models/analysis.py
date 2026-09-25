import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, JSON, DateTime, Float
from app.db.session import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    input_type = Column(String(20), nullable=False)  # 'url', 'text', 'image', 'video'
    input_reference = Column(String(1000), nullable=False)  # URL or original filename
    authenticity_score = Column(Integer, nullable=False)  # 0 to 100
    confidence_percentage = Column(Integer, nullable=False)  # 0 to 100
    risk_level = Column(String(10), nullable=False)  # 'Low', 'Medium', 'High'
    
    # JSON columns to store module details and text descriptions
    # Maps to JSONB on PostgreSQL and Text on SQLite automatically
    module_results = Column(JSON, nullable=False)
    explanation = Column(JSON, nullable=False)  # list of strings
    processing_time = Column(Float, default=0.0, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        modules_run = []
        if isinstance(self.module_results, dict):
            if self.module_results.get("text_nlp") is not None:
                modules_run.append("text_nlp")
            if self.module_results.get("image_forensics") is not None:
                modules_run.append("image_forensics")
            if self.module_results.get("deepfake") is not None:
                modules_run.append("deepfake")

        # Import compute_final_decision inside to_dict to prevent circular imports
        from app.fusion.fusion_engine import compute_final_decision
        
        # Determine if we need to reduce confidence
        reduce_confidence = False
        if isinstance(self.module_results, dict):
            text_nlp = self.module_results.get("text_nlp")
            if text_nlp and isinstance(text_nlp, dict):
                reduce_confidence = text_nlp.get("reduce_confidence", False)

        # Generate the unified Final Decision Object dynamically
        decision = compute_final_decision(
            input_type=self.input_type,
            module_results=self.module_results,
            reduce_confidence=reduce_confidence
        )

        supporting_sources = []
        for ev in decision["supporting_evidence"]:
            supporting_sources.append({
                "name": f"{ev.get('source_type', 'Source')}: {ev.get('publisher', 'Publisher')}",
                "url": ev.get("url", "")
            })

        return {
            "analysis_id": self.id,
            "input_type": self.input_type,
            "input_reference": self.input_reference,
            "authenticity_score": self.authenticity_score,
            "confidence_percentage": self.confidence_percentage,
            "risk_level": self.risk_level,
            "modules_run": modules_run,
            "module_results": self.module_results,
            "processing_time": self.processing_time or 0.0,
            
            # Centralized Decision Engine Fields
            "verdict": decision["verdict"],
            "recommendation": decision["recommendation"],
            "explanation": decision["explanation"],
            "supporting_evidence": decision["supporting_evidence"],
            "contradicting_evidence": decision["contradicting_evidence"],
            "confidence": decision["confidence"],
            "evidence_sources": decision["evidence_sources"],
            "primary_entity": decision["primary_entity"],
            "named_entities": decision["named_entities"],
            "supporting_sources": supporting_sources,
            "created_at": self.created_at.isoformat() + "Z"
        }
