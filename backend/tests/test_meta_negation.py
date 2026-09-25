import pytest
from app.analysis.nlp_module import parse_meta_statement

def test_parse_meta_statement_compound_precedence():
    # Compound negation "not false" -> TRUE (identity)
    inner, op = parse_meta_statement("It is not false that the Earth orbits the Sun.")
    assert inner.rstrip(".") == "the Earth orbits the Sun"
    assert op == "TRUE"
    
    inner, op = parse_meta_statement("It is not false that the Earth is flat.")
    assert inner.rstrip(".") == "the Earth is flat"
    assert op == "TRUE"

    # Compound negation "not true" -> FALSE (negation)
    inner, op = parse_meta_statement("It is not true that the Earth is flat.")
    assert inner.rstrip(".") == "the Earth is flat"
    assert op == "FALSE"

    inner, op = parse_meta_statement("It is not true that the Earth orbits the Sun.")
    assert inner.rstrip(".") == "the Earth orbits the Sun"
    assert op == "FALSE"

    # Single-word operators
    inner, op = parse_meta_statement("It is false that the Earth orbits the Sun.")
    assert inner.rstrip(".") == "the Earth orbits the Sun"
    assert op == "FALSE"

    inner, op = parse_meta_statement("It is true that the Earth orbits the Sun.")
    assert inner.rstrip(".") == "the Earth orbits the Sun"
    assert op == "TRUE"

    # Standard claims
    inner, op = parse_meta_statement("The Earth orbits the Sun.")
    assert op is None

    inner, op = parse_meta_statement("The Earth is completely flat.")
    assert op is None

def test_meta_negation_not_false_earth_orbits_sun(client):
    """1. It is not false that the Earth orbits the Sun. -> Expected: TRUE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "It is not false that the Earth orbits the Sun."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "TRUE"
    assert data["authenticity_score"] >= 80

def test_meta_negation_not_false_earth_is_flat(client):
    """2. It is not false that the Earth is flat. -> Expected: FALSE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "It is not false that the Earth is flat."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "FALSE"
    assert data["authenticity_score"] < 40

def test_meta_negation_not_true_earth_is_flat(client):
    """3. It is not true that the Earth is flat. -> Expected: TRUE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "It is not true that the Earth is flat."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "TRUE"
    assert data["authenticity_score"] >= 80

def test_meta_negation_not_true_earth_orbits_sun(client):
    """4. It is not true that the Earth orbits the Sun. -> Expected: FALSE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "It is not true that the Earth orbits the Sun."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "FALSE"
    assert data["authenticity_score"] < 40

def test_meta_negation_false_earth_orbits_sun(client):
    """5. It is false that the Earth orbits the Sun. -> Expected: FALSE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "It is false that the Earth orbits the Sun."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "FALSE"
    assert data["authenticity_score"] < 40

def test_meta_affirmation_true_earth_orbits_sun(client):
    """6. It is true that the Earth orbits the Sun. -> Expected: TRUE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "It is true that the Earth orbits the Sun."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "TRUE"
    assert data["authenticity_score"] >= 80

def test_normal_claim_earth_orbits_sun(client):
    """7. Existing normal claim: The Earth orbits the Sun. -> Expected: TRUE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "The Earth orbits the Sun."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "TRUE"
    assert data["authenticity_score"] >= 80

def test_normal_claim_earth_is_flat(client):
    """8. Existing normal false claim: The Earth is completely flat. -> Expected: FALSE"""
    res = client.post("/api/analyze", json={"input_type": "text", "text": "The Earth is completely flat."})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "FALSE"
    assert data["authenticity_score"] < 40
