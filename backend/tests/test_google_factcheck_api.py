import pytest
from unittest.mock import patch, MagicMock
import httpx
from app.analysis.nlp_module import query_google_factcheck_api
from app.core.config import settings

def test_query_google_factcheck_api_missing_key(monkeypatch):
    """Verify behavior when GOOGLE_FACTCHECK_API_KEY is not set."""
    monkeypatch.setattr(settings, "GOOGLE_FACTCHECK_API_KEY", None)
    claims, query, req_url, body, status, explanation = query_google_factcheck_api("The Earth is flat")
    
    assert claims == []
    assert status == "API key missing"
    assert "not configured" in explanation.lower()

def test_query_google_factcheck_api_success_results(monkeypatch):
    """Verify behavior when API returns matching claims."""
    monkeypatch.setattr(settings, "GOOGLE_FACTCHECK_API_KEY", "MOCK_KEY_FOR_TESTING")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "claims": [
            {
                "text": "The Earth is flat.",
                "claimant": "Social Media",
                "claimReview": [
                    {
                        "publisher": {"name": "Full Fact", "site": "fullfact.org"},
                        "textualRating": "False",
                        "title": "The Earth is roughly spherical, not flat",
                        "url": "https://fullfact.org/online/earth-flat/"
                    }
                ]
            }
        ]
    }
    mock_response.text = '{"claims": [...]}'
    
    with patch("httpx.get", return_value=mock_response):
        claims, query, req_url, body, status, explanation = query_google_factcheck_api("The Earth is completely flat.")
        
        assert len(claims) == 1
        assert claims[0]["text"] == "The Earth is flat."
        assert status == "Verified claim found"
        assert "MOCK_KEY_FOR_TESTING" not in req_url  # Ensure key is not exposed in log query string

def test_query_google_factcheck_api_no_results(monkeypatch):
    """Verify behavior when API returns 200 OK but no matching claims (empty list)."""
    monkeypatch.setattr(settings, "GOOGLE_FACTCHECK_API_KEY", "MOCK_KEY_FOR_TESTING")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"claims": []}
    mock_response.text = '{"claims": []}'
    
    with patch("httpx.get", return_value=mock_response):
        claims, query, req_url, body, status, explanation = query_google_factcheck_api("Quantum xylophones resonant frequency 8472931057421.")
        
        assert claims == []
        assert status == "No verified claim found"

def test_query_google_factcheck_api_invalid_key(monkeypatch):
    """Verify behavior when API key is invalid (400 or 403 error)."""
    monkeypatch.setattr(settings, "GOOGLE_FACTCHECK_API_KEY", "INVALID_MOCK_KEY")
    
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT"
        }
    }
    mock_response.text = '{"error": {"message": "API key not valid"}}'
    
    with patch("httpx.get", return_value=mock_response):
        claims, query, req_url, body, status, explanation = query_google_factcheck_api("The Earth is flat.")
        
        assert claims == []
        assert status == "Invalid API key"

def test_query_google_factcheck_api_quota_exceeded(monkeypatch):
    """Verify behavior when API rate limit or quota is exceeded (429 error)."""
    monkeypatch.setattr(settings, "GOOGLE_FACTCHECK_API_KEY", "MOCK_KEY")
    
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = '{"error": {"message": "Quota exceeded"}}'
    
    with patch("httpx.get", return_value=mock_response):
        claims, query, req_url, body, status, explanation = query_google_factcheck_api("The Earth is flat.")
        
        assert claims == []
        assert status == "Quota exceeded"

def test_query_google_factcheck_api_timeout(monkeypatch):
    """Verify behavior on network timeout."""
    monkeypatch.setattr(settings, "GOOGLE_FACTCHECK_API_KEY", "MOCK_KEY")
    
    with patch("httpx.get", side_effect=httpx.TimeoutException("Timeout error")):
        claims, query, req_url, body, status, explanation = query_google_factcheck_api("The Earth is flat.")
        
        assert claims == []
        assert status == "Network timeout"
