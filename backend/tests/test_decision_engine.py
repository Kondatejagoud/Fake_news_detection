import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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

def check_no_contradiction(data: dict):
    """
    Programmatically asserts that no contradictory combination of
    verdict, risk level, and authenticity score exists.
    """
    auth = data["authenticity_score"]
    verdict = data["verdict"]
    risk = data["risk_level"]
    
    # Forbidden rules
    # 1. Likely Authentic/True must be Low Risk and score >= 80
    if verdict in ["Likely Authentic", "Likely True"]:
        assert risk == "Low", f"Contradiction: Verdict is '{verdict}' but Risk is '{risk}'"
        assert auth >= 80, f"Contradiction: Verdict is '{verdict}' but Authenticity is {auth}%"
        
    # 2. Needs Review / Suspicious must be Medium Risk and score 40-79
    if verdict in ["Needs Review", "Suspicious"]:
        assert risk == "Medium", f"Contradiction: Verdict is '{verdict}' but Risk is '{risk}'"
        assert auth >= 40 and auth < 80, f"Contradiction: Verdict is '{verdict}' but Authenticity is {auth}%"
        
    # 3. Likely Manipulated/False must be High Risk and score < 40
    if verdict in ["Likely Manipulated", "Likely False"]:
        assert risk == "High", f"Contradiction: Verdict is '{verdict}' but Risk is '{risk}'"
        assert auth < 40, f"Contradiction: Verdict is '{verdict}' but Authenticity is {auth}%"

def test_claim_modi_resigned(client):
    payload = {
        "input_type": "text",
        "text": "Narendra Modi resigned"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Expected: Needs Review or Likely False, definitely NOT Likely True
    assert data["verdict"] in ["Needs Review", "Likely False", "Suspicious"]
    assert data["verdict"] not in ["Likely True", "Likely Authentic"]
    check_no_contradiction(data)

def test_claim_isro_chandrayaan3(client):
    payload = {
        "input_type": "text",
        "text": "ISRO launched Chandrayaan-3"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Expected: Likely True
    assert data["verdict"] in ["Likely True", "Likely Authentic"]
    assert data["authenticity_score"] >= 80
    check_no_contradiction(data)

def test_claim_earth_flat(client):
    payload = {
        "input_type": "text",
        "text": "The Earth is flat"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Expected: Likely False
    assert data["verdict"] in ["Likely False", "Likely Manipulated"]
    assert data["authenticity_score"] < 40
    check_no_contradiction(data)
