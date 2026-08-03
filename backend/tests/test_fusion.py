import pytest
from app.fusion.fusion_engine import fuse_results, calculate_disagreement_penalty
from app.explanation.report_generator import generate_report

def test_calculate_disagreement_penalty():
    # Single score: no penalty
    assert calculate_disagreement_penalty([0.5]) == 0.0
    # No scores: no penalty
    assert calculate_disagreement_penalty([]) == 0.0
    # Equal scores: no penalty
    assert calculate_disagreement_penalty([0.5, 0.5]) == 0.0
    # Complete disagreement (0.0 and 1.0): maximum penalty (30.0%)
    assert calculate_disagreement_penalty([0.0, 1.0]) == 30.0
    # Medium disagreement
    assert calculate_disagreement_penalty([0.2, 0.7]) == pytest.approx(15.0)

def test_fuse_results_single_source():
    # Test text NLP only
    auth, conf, risk = fuse_results(text_nlp_score=0.2)
    assert auth == 80  # (1 - 0.2) * 100
    assert conf == 75  # Base confidence for 1 module
    assert risk == "Low"

    # Test high fake probability
    auth, conf, risk = fuse_results(text_nlp_score=0.9)
    assert auth == 10
    assert conf == 75
    assert risk == "High"

def test_fuse_results_multi_source():
    # Two modules in agreement
    auth, conf, risk = fuse_results(text_nlp_score=0.1, image_forensics_score=0.2)
    # Weighted average: (0.1*0.35 + 0.2*0.4) / (0.35+0.4) = (0.035+0.08) / 0.75 = 0.115/0.75 = 0.1533
    # Authenticity: round((1 - 0.1533) * 100) = 85
    assert auth == 85
    # Base confidence = 88. Disagreement = 0.1, penalty = 3. Conf = 85
    assert conf == 85
    assert risk == "Low"

    # Three modules in extreme disagreement
    auth, conf, risk = fuse_results(text_nlp_score=0.1, image_forensics_score=0.9, deepfake_score=0.5)
    # Weighted avg: (0.1*0.35 + 0.9*0.4 + 0.5*0.45) / 1.2 = (0.035 + 0.36 + 0.225) / 1.2 = 0.62 / 1.2 = 0.5167
    # Authenticity = round((1 - 0.5167) * 100) = 48
    assert auth == 48
    assert risk == "Medium"
    # Base confidence = 96. Disagreement = 0.8, penalty = 24. Conf = 72
    assert conf == 72

def test_generate_report():
    module_results = {
        "text_nlp": {"score": 0.8, "flagged_claims": ["claim1", "claim2"]},
        "image_forensics": None,
        "deepfake": None
    }
    report = generate_report(
        input_type="text",
        authenticity_score=20,
        confidence_percentage=75,
        risk_level="High",
        module_results=module_results
    )
    assert len(report) >= 3
    # Verify citations are present
    assert any("text" in r.lower() for r in report)
    assert any("nlp" in r.lower() for r in report)
    assert any("high" in r.lower() or "warning" in r.lower() for r in report)
