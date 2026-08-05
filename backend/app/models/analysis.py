import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, JSON, DateTime
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
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        modules_run = []
        if isinstance(self.module_results, dict):
            if self.module_results.get("text_nlp") is not None:
                modules_run.append("text_nlp")
            if self.module_results.get("image_forensics") is not None:
                modules_run.append("image_forensics")
            if self.module_results.get("deepfake") is not None:
                modules_run.append("deepfake")

        # Extract final decision details if present, otherwise compute dynamically for backward compatibility
        final_dec = {}
        if isinstance(self.module_results, dict):
            final_dec = self.module_results.get("final_decision", {})
        
        verdict = final_dec.get("verdict")
        supporting_sources = final_dec.get("supporting_sources", [])

        if not verdict:
            if self.authenticity_score >= 80:
                verdict = "Likely Authentic" if self.input_type in ["image", "video"] else "Likely True"
            elif self.authenticity_score >= 65:
                verdict = "Needs Review"
            elif self.authenticity_score >= 40:
                verdict = "Suspicious"
            else:
                verdict = "Likely Manipulated" if self.input_type in ["image", "video"] else "Likely False"

        if not supporting_sources and isinstance(self.module_results, dict):
            text_nlp = self.module_results.get("text_nlp")
            if text_nlp and isinstance(text_nlp, dict):
                debug = text_nlp.get("factcheck_debug", {})
                if debug and isinstance(debug, dict):
                    evidence_list = debug.get("evidence_list", [])
                    for ev in evidence_list:
                        if isinstance(ev, dict) and ev.get("url"):
                            supporting_sources.append({
                                "name": f"{ev.get('source_type', 'Source')}: {ev.get('publisher', 'Publisher')}",
                                "url": ev.get("url")
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
            "explanation": self.explanation,
            "verdict": verdict,
            "supporting_sources": supporting_sources,
            "created_at": self.created_at.isoformat() + "Z"
        }
