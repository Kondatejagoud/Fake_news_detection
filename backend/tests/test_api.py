import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from app.main import app
from app.db.session import Base, get_db
from app.models.analysis import Analysis

# Create in-memory database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(name="db_session")
def fixture_db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(name="client")
def fixture_client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_text_json(client):
    payload = {
        "input_type": "text",
        "text": "This is a sample statement to analyze for fake claims."
    }
    with patch("app.api.routes.analyze.analyze_text") as mock_analyze:
        mock_analyze.return_value = {
            "score": 0.10,
            "flagged_claims": [],
            "factcheck_debug": {
                "user_claim": "This is a sample statement to analyze for fake claims.",
                "matched_claim": "No matching verified claim found.",
                "verdict": "Unverified",
                "evidence_list": []
            },
            "factcheck_matched": False,
            "reduce_confidence": False
        }
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["input_type"] == "text"
        assert data["authenticity_score"] == 90  # Real keyword model returns 0.1
        assert data["risk_level"] == "Low"
        assert "text_nlp" in data["module_results"]

def test_analyze_url_json(client):
    payload = {
        "input_type": "url",
        "url": "https://some-news-website.com/article-1"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert data["input_type"] == "url"
    assert data["input_reference"] == "https://some-news-website.com/article-1"

def test_analyze_invalid_input_type(client):
    payload = {
        "input_type": "invalid_type",
        "text": "hello"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 400
