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
    if verdict == "TRUE":
        assert risk == "Low", f"Contradiction: Verdict is '{verdict}' but Risk is '{risk}'"
        assert auth >= 80, f"Contradiction: Verdict is '{verdict}' but Authenticity is {auth}%"
        
    if verdict == "NEEDS REVIEW":
        assert risk == "Medium", f"Contradiction: Verdict is '{verdict}' but Risk is '{risk}'"
        assert auth >= 65 and auth < 80, f"Contradiction: Verdict is '{verdict}' but Authenticity is {auth}%"
        
    if verdict == "UNVERIFIED":
        assert risk == "Medium", f"Contradiction: Verdict is '{verdict}' but Risk is '{risk}'"
        assert auth >= 40 and auth < 65, f"Contradiction: Verdict is '{verdict}' but Authenticity is {auth}%"
        
    if verdict == "FALSE":
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
    
    # Expected: NEEDS REVIEW, FALSE, or UNVERIFIED, definitely NOT TRUE
    assert data["verdict"] in ["NEEDS REVIEW", "FALSE", "UNVERIFIED"]
    assert data["verdict"] != "TRUE"
    check_no_contradiction(data)

def test_claim_isro_chandrayaan3(client):
    payload = {
        "input_type": "text",
        "text": "ISRO launched Chandrayaan-3"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Expected: TRUE
    assert data["verdict"] == "TRUE"
    assert data["authenticity_score"] >= 80
    check_no_contradiction(data)

def test_claim_earth_flat(client):
    payload = {
        "input_type": "text",
        "text": "The Earth is completely flat."
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "FALSE"
    assert data["authenticity_score"] < 40
    check_no_contradiction(data)

def test_claim_earth_orbits_sun(client):
    payload = {
        "input_type": "text",
        "text": "The Earth orbits the Sun."
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "TRUE"
    assert data["authenticity_score"] >= 80
    check_no_contradiction(data)

def test_claim_water_freezes(client):
    payload = {
        "input_type": "text",
        "text": "Water freezes at 0 degrees Celsius at standard atmospheric pressure."
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ["TRUE", "NEEDS REVIEW"]
    check_no_contradiction(data)

def test_claim_breathe_in_vacuum(client):
    payload = {
        "input_type": "text",
        "text": "Humans can breathe normally in pure vacuum without oxygen."
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ["FALSE", "NEEDS REVIEW", "UNVERIFIED"]
    assert data["verdict"] != "TRUE"
    check_no_contradiction(data)

def test_claim_obscure_unsupported(client):
    payload = {
        "input_type": "text",
        "text": "Quantum xylophones resonant frequency 8472931057421."
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "UNVERIFIED"
    assert data["confidence_percentage"] <= 40
    check_no_contradiction(data)
