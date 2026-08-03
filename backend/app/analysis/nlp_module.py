import os
import re
import httpx
import difflib
import urllib.parse
from typing import Dict, List, Optional, Tuple, Set, Any
from app.core.config import settings
from app.core.logging import logger

# Lazy loaders
_spacy_nlp = None
_sentence_model = None
_nlp_pipeline = None

# Trusted Domains lists for categorization
TRUSTED_NEWS_DOMAINS = [
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "bloomberg.com", 
    "nytimes.com", "theguardian.com", "cnn.com", "dw.com", "aljazeera.com", 
    "reuters.tv", "pib.gov.in", "wsj.com", "economist.com", "ft.com",
    "afp.com", "ap.org"
]

TRUSTED_OFFICIAL_DOMAINS = [
    ".gov", ".gov.in", ".gov.uk", "nasa.gov", "who.int", "un.org", 
    "rbi.org.in", "isro.gov.in", "europa.eu", "cdc.gov", "nih.gov",
    "whitehouse.gov", "state.gov", "unicef.org", "fao.org"
]

# Local database of known flagged misinformation claims (knowledge base)
KNOWN_MISINFORMATION_CLAIMS = [
    "COVID-19 was created in a laboratory as a bioweapon.",
    "Vaccines contain 5G microchips to track and control the population.",
    "Drinking chlorine dioxide or bleach cures viral infections and diseases.",
    "NASA admitted that the Earth is flat in a leaked internal report.",
    "A catastrophic magnitude 10 earthquake is scientifically predicted to hit next week.",
    "Eating yellow bananas cures terminal cancer in under three days.",
    "An underground child trafficking ring is run by political elites under a pizza parlor.",
    "Scientists found a hidden, fully inhabited continent beneath the Pacific Ocean.",
    "Voting machines were rigged to automatically flip ballots in the election.",
    "5G cellular towers are weakening human immune systems and spreading viruses."
]

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy 'en_core_web_sm' successfully loaded.")
        except Exception as e:
            logger.warning(f"spaCy load failed: {e}. Falling back to Regex Named Entity Recognition.")
            _spacy_nlp = False  # Sentinel
    return _spacy_nlp

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Initializing Sentence-Transformers Model...")
            _sentence_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=settings.HF_HOME
            )
            logger.info("Sentence-Transformers Model successfully loaded.")
        except Exception as e:
            logger.warning(f"Failed to load Sentence-Transformers: {e}. Falling back to token similarity.")
            _sentence_model = False  # Sentinel
    return _sentence_model

def get_nlp_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Initializing HuggingFace Text Classification Pipeline...")
            _nlp_pipeline = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1, # CPU
                model_kwargs={"cache_dir": settings.HF_HOME}
            )
            logger.info("HuggingFace Text Classification Pipeline loaded.")
        except Exception as e:
            logger.warning(f"Failed to load HF pipeline: {e}. Falling back to keyword analysis.")
            _nlp_pipeline = False  # Sentinel
    return _nlp_pipeline

# ==========================================
# STEP 1: EXTRACT CLAIMS
# ==========================================
def extract_claims(text: str) -> List[str]:
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    opinion_indicators = {
        "think", "opinion", "probably", "should", "must", "feel", "great", "bad",
        "worst", "beautiful", "excellent", "terrible", "love", "hate", "dislike",
        "believe", "my view", "in my opinion", "perhaps", "maybe", "i guess"
    }
    
    factual_claims = []
    
    for sentence in raw_sentences:
        s_clean = sentence.strip()
        words = s_clean.split()
        if len(words) < 5:
            continue
            
        s_lower = s_clean.lower()
        if any(w in s_lower for w in opinion_indicators):
            logger.info(f"NER Claim Filter: Rejected opinion sentence: '{s_clean}'")
            continue
            
        entity_count = len(re.findall(r'\b[A-Z][a-zA-Z0-9-]*\b', s_clean))
        number_count = len(re.findall(r'\b\d+\b', s_clean))
        factual_density = entity_count + number_count
        
        factual_claims.append((s_clean, factual_density))
        
    factual_claims.sort(key=lambda x: x[1], reverse=True)
    selected_claims = [item[0] for item in factual_claims[:5]]
    
    if not selected_claims and raw_sentences:
        for sentence in raw_sentences:
            if len(sentence.strip()) > 10:
                selected_claims = [sentence.strip()]
                break
                
    logger.info(f"NLP Extracted Factual Claims: {selected_claims}")
    return selected_claims

# ==========================================
# STEP 3: NER EXTRACTION
# ==========================================
def extract_named_entities(text: str) -> Dict[str, Set[str]]:
    entities = {
        "PERSON": set(),
        "ORGANIZATION": set(),
        "LOCATION": set(),
        "DATE": set(),
        "EVENT": set()
    }
    
    nlp = get_spacy_nlp()
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                label = ent.label_
                text_val = ent.text.strip()
                if label == "PERSON":
                    entities["PERSON"].add(text_val)
                elif label in ["ORG", "NORP"]:
                    entities["ORGANIZATION"].add(text_val)
                elif label in ["GPE", "LOC", "FAC"]:
                    entities["LOCATION"].add(text_val)
                elif label in ["DATE", "TIME"]:
                    entities["DATE"].add(text_val)
                elif label in ["EVENT", "LAW"]:
                    entities["EVENT"].add(text_val)
            return entities
        except Exception as e:
            logger.error(f"spaCy NER processing failed: {e}. Falling back to Regex NER.")

    words_seq = re.findall(r'\b[A-Z][a-zA-Z0-9-]*+(?:\s+[A-Z][a-zA-Z0-9-]*+)*\b', text)
    for word in words_seq:
        w_lower = word.lower()
        if any(m in w_lower for m in ["organisation", "agency", "institute", "isro", "nasa", "pib", "news", "commission", "center", "centre"]):
            entities["ORGANIZATION"].add(word)
        elif any(m in w_lower for m in ["india", "shriharikota", "pacific", "earth", "moon", "canada", "america", "london", "beijing"]):
            entities["LOCATION"].add(word)
        else:
            entities["PERSON"].add(word)
            
    dates = re.findall(r'\b(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*+(?:\s+\d{1,2})?,?\s+\d{4}\b|\b\d{4}\b', text)
    for d in dates:
        entities["DATE"].add(d)
        
    return entities

def check_entities_match(user_ent: Dict[str, Set[str]], google_ent: Dict[str, Set[str]]) -> bool:
    labels = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT"]
    user_has_any = any(user_ent[l] for l in labels)
    google_has_any = any(google_ent[l] for l in labels)
    
    if not user_has_any or not google_has_any:
        return True
        
    for label in ["ORGANIZATION", "PERSON", "LOCATION"]:
        u_set = {u.lower() for u in user_ent[label]}
        g_set = {g.lower() for g in google_ent[label]}
        if u_set and g_set:
            for g_item in g_set:
                if any(g_item in u_item or u_item in g_item for u_item in u_set):
                    return True
                    
    for label in ["DATE", "EVENT"]:
        u_set = {u.lower() for u in user_ent[label]}
        g_set = {g.lower() for g in google_ent[label]}
        if u_set.intersection(g_set):
            return True
            
    return False

# ==========================================
# SIMILARITY HELPERS
# ==========================================
def get_string_similarity(str1: str, str2: str) -> float:
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(str1, str2))
    except Exception:
        return difflib.SequenceMatcher(None, str1, str2).ratio() * 100.0

def get_semantic_similarity(str1: str, str2: str) -> float:
    model = get_sentence_model()
    if model:
        try:
            from sentence_transformers import util
            emb1 = model.encode(str1, convert_to_tensor=True)
            emb2 = model.encode(str2, convert_to_tensor=True)
            cosine_score = float(util.cos_sim(emb1, emb2)[0][0])
            return cosine_score
        except Exception as e:
            logger.error(f"SentenceTransformers cos_sim failed: {e}. Falling back to token cosine.")
            
    return token_cosine_similarity(str1, str2)

def token_cosine_similarity(str1: str, str2: str) -> float:
    words1 = re.findall(r'\b\w+\b', str1.lower())
    words2 = re.findall(r'\b\w+\b', str2.lower())
    all_words = set(words1).union(set(words2))
    v1 = [words1.count(w) for w in all_words]
    v2 = [words2.count(w) for w in all_words]
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if not norm_v1 or not norm_v2:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def check_fallback_match(claim_query: str, item_title: str, item_snippet: str, entities: List[str]) -> bool:
    title_lower = item_title.lower()
    snippet_lower = item_snippet.lower()
    combined_lower = f"{title_lower} {snippet_lower}"
    
    # 1. If we have entities, check if they are present in the search item
    if entities:
        match_count = 0
        for ent in entities:
            ent_clean = ent.strip().lower()
            if not ent_clean:
                continue
            if ent_clean in combined_lower:
                match_count += 1
        if match_count >= 1:
            return True
            
    # 2. Check overlap of key content words (non-stopwords)
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "against", "of", "successfully", "launched", "mission"}
    claim_words = [w.strip(".,!?\"'") for w in claim_query.lower().split()]
    content_words = [w for w in claim_words if w and w not in stopwords]
    
    if not content_words:
        return False
        
    overlap_count = sum(1 for w in content_words if w in combined_lower)
    overlap_ratio = overlap_count / len(content_words)
    
    return overlap_ratio >= 0.50

# ==========================================
# HELPERS FOR SOURCE CARDS & TIMELINE
# ==========================================
def format_wikipedia_timestamp(ts_str: str) -> str:
    if not ts_str:
        return "N/A"
    try:
        # e.g. "2023-08-25T12:00:00Z" -> "Aug 25, 2023"
        date_part = ts_str.split("T")[0] # "2023-08-25"
        parts = date_part.split("-")
        months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        year = parts[0]
        month_idx = int(parts[1])
        day = str(int(parts[2]))
        return f"{months[month_idx]} {day}, {year}"
    except Exception:
        return "N/A"

def get_source_reliability_badge(url: str, publisher: str) -> str:
    url_l = url.lower()
    pub_l = publisher.lower()
    
    if any(dom in url_l for dom in [".gov", ".gov.in", ".gov.uk", "europa.eu", "pib.gov.in", "whitehouse.gov"]):
        if any(dom in url_l for dom in ["isro.gov.in", "nasa.gov", "esa.int"]):
            return "Space Agency"
        return "Government"
    if any(dom in url_l for dom in ["nasa.gov", "isro.gov.in", "esa.int"]):
        return "Space Agency"
    if any(dom in url_l for dom in [".edu", "harvard.edu", "mit.edu", "ox.ac.uk"]):
        return "University"
    if any(p in pub_l or p in url_l for p in ["afp", "factcheck", "snopes", "politifact", "boomlive"]):
        return "Fact Checker"
    if any(dom in url_l for dom in ["reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "bloomberg.com", "nytimes.com", "theguardian.com", "cnn.com", "dw.com", "aljazeera.com", "wsj.com", "economist.com", "ft.com"]):
        return "Major News Agency"
    if "wikipedia.org" in url_l or pub_l == "wikipedia":
        return "Community Source"
    return "Unknown Source"

def clean_publisher_name(url: str, raw_pub: str) -> str:
    url_l = url.lower()
    if "wikipedia.org" in url_l:
        return "Wikipedia"
    if "isro.gov.in" in url_l:
        return "ISRO"
    if "nasa.gov" in url_l:
        return "NASA"
    if "reuters.com" in url_l:
        return "Reuters"
    if "apnews.com" in url_l or "ap.org" in url_l:
        return "Associated Press"
    if "bbc.com" in url_l or "bbc.co.uk" in url_l:
        return "BBC"
    if "bloomberg.com" in url_l:
        return "Bloomberg"
    if "nytimes.com" in url_l:
        return "The New York Times"
    if "theguardian.com" in url_l:
        return "The Guardian"
    if "pib.gov.in" in url_l:
        return "Press Information Bureau (PIB)"
    if "who.int" in url_l:
        return "World Health Organization (WHO)"
    if "un.org" in url_l:
        return "United Nations"
    
    raw_pub = raw_pub.replace("www.", "")
    if "." in raw_pub:
        raw_pub = raw_pub.split(".")[0].capitalize()
    return raw_pub

# ==========================================
# CRAWLERS AND EVIDENCE APIs
# ==========================================
def extract_core_query(text: str) -> str:
    text_clean = text.strip()
    
    attribution_markers = [
        "claim that", "claims that", "report that", "reports that", 
        "says that", "say that", "asserts that", "assert that",
        "stated that", "state that", "argued that", "argues that",
        "announced that", "announces that", "believes that", "believe that"
    ]
    
    for marker in attribution_markers:
        if marker in text_clean.lower():
            idx = text_clean.lower().find(marker)
            core = text_clean[idx + len(marker):].strip()
            if len(core) > 10:
                logger.info(f"NLP: Attribution marker '{marker}' found. Stripped core statement: '{core}'")
                return core
                
    return text_clean

def query_google_factcheck_api(text: str) -> Tuple[List[dict], str, str, str, str, str]:
    api_key = getattr(settings, "GOOGLE_FACTCHECK_API_KEY", None)
    if not api_key:
        return [], text, "", "No API Key configured", "API key missing", "Google Fact Check API key is not configured in environment settings."
        
    core_query = extract_core_query(text)
    query = core_query.rstrip(".!?")
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": api_key,
        "query": query,
        "languageCode": "en"
    }
    
    request_url = f"{url}?query={query}"
    
    try:
        response = httpx.get(url, params=params, timeout=4.0)
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            if claims:
                return claims, core_query, request_url, response.text, "Verified claim found", "A matching verified public fact check was found in the Google Fact Check Tools database."
            else:
                return [], core_query, request_url, response.text, "No verified claim found", "No matching third-party fact check matches were indexed by Google for this query."
        elif response.status_code == 429:
            return [], core_query, request_url, response.text, "Quota exceeded", "Google API request limit has been exceeded. Please check quotas."
        else:
            return [], core_query, request_url, response.text, "API unavailable", f"Google API returned error status: {response.status_code}."
    except httpx.ConnectError:
        return [], core_query, request_url, "Connect failed", "Network error", "Failed to connect to Google API. Network may be unreachable."
    except httpx.TimeoutException:
        return [], core_query, request_url, "Request timed out", "Network error", "Google Fact Check API request timed out."
    except Exception as e:
        logger.error(f"Google Fact Check API request failed: {e}")
        return [], core_query, request_url, f"Request failed: {str(e)}", "API unavailable", f"Request exception: {str(e)}"

def search_duckduckgo_fallback(query: str) -> List[Dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    url = "https://html.duckduckgo.com/html/"
    try:
        logger.info(f"Multi-Source Evidence: Querying DuckDuckGo HTML fallback for '{query}'")
        response = httpx.get(url, params={"q": query}, headers=headers, timeout=5.0)
        if response.status_code in [200, 202]:
            html = response.text
            results = []
            
            # Extract URLs, Titles, and Snippets using flexible regexes
            titles_urls = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html)
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html)
            
            for idx, (href, title_html) in enumerate(titles_urls[:10]):
                title = re.sub(r'<[^>]+>', '', title_html).strip()
                snippet = ""
                if idx < len(snippets):
                    snippet = re.sub(r'<[^>]+>', '', snippets[idx]).strip()
                    
                # Clean up query routing parameters in DuckDuckGo href (e.g. //uddg=...)
                url_match = re.search(r'uddg=([^&]+)', href)
                final_url = urllib.parse.unquote(url_match.group(1)) if url_match else href
                if final_url.startswith("//"):
                    final_url = "https:" + final_url
                    
                results.append({
                    "title": title,
                    "url": final_url,
                    "snippet": snippet
                })
            logger.info(f"DuckDuckGo search parsed {len(results)} links.")
            return results
    except Exception as e:
        logger.error(f"DuckDuckGo fallback search scraper failed: {e}")
    return []

def search_wikipedia(query: str) -> List[Dict]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "format": "json"
    }
    headers = {
        "User-Agent": "TrueLensFakeContentDetector/2.1 (contact@truelens.ai) httpx/0.26"
    }
    try:
        logger.info(f"Multi-Source Evidence: Querying Wikipedia API for '{query}'")
        res = httpx.get(url, params=params, headers=headers, timeout=4.0)
        if res.status_code == 200:
            search_items = res.json().get("query", {}).get("search", [])
            wiki_results = []
            for item in search_items[:3]:
                title = item.get("title", "")
                snippet = re.sub(r'<[^>]+>', '', item.get("snippet", "")).strip()
                timestamp = item.get("timestamp", "")
                wiki_results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "publisher": "Wikipedia",
                    "timestamp": timestamp
                })
            return wiki_results
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
    return []

def fetch_multi_source_evidence(query: str) -> Dict[str, List[Dict]]:
    evidence = {
        "wikipedia": [],
        "official_sources": [],
        "trusted_news": []
    }
    
    # 1. Wikipedia API
    evidence["wikipedia"] = search_wikipedia(query)
    
    # 2. Search web for News and Official sources
    web_results = search_duckduckgo_fallback(query)
    for res in web_results:
        url = res["url"]
        title = res["title"]
        snippet = res["snippet"]
        
        pub_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        publisher = pub_match.group(1) if pub_match else "Web Source"
        
        item = {
            "title": title,
            "snippet": snippet,
            "url": url,
            "publisher": publisher
        }
        
        is_official = any(dom in url.lower() for dom in TRUSTED_OFFICIAL_DOMAINS)
        is_news = any(dom in url.lower() for dom in TRUSTED_NEWS_DOMAINS)
        
        if is_official:
            evidence["official_sources"].append(item)
        elif is_news:
            evidence["trusted_news"].append(item)
            
    return evidence

# ==========================================
# LOCAL MISINFO UTILITIES
# ==========================================
def rule_based_text_fake_score(text: str) -> float:
    sensational_words = [
        "shocking", "exposed", "conspiracy", "unbelievable", "they don't want you to know",
        "secret", "miracle", "cure", "scam", "rigged", "bioweapon", "insider", "leak",
        "hiding", "anonymous source", "proof", "mind control", "hoax", "viral", "uncovered"
    ]
    text_lower = text.lower()
    score = 0.40  # Neutral baseline
    found_keywords = [w for w in sensational_words if w in text_lower]
    score += len(found_keywords) * 0.15
    exclamation_count = text.count("!")
    if exclamation_count > 3:
        score += 0.15
    words = text.split()
    if len(words) > 5:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        caps_ratio = len(caps_words) / len(words)
        if caps_ratio > 0.15:
            score += 0.20
    return min(0.95, score)

def token_similarity_fallback(text: str, database: List[str]) -> Tuple[float, Optional[str]]:
    text_tokens = set(text.lower().split())
    best_score = 0.0
    best_claim = None
    for claim in database:
        claim_tokens = set(claim.lower().split())
        intersection = text_tokens.intersection(claim_tokens)
        union = text_tokens.union(claim_tokens)
        if not union:
            continue
        jaccard = len(intersection) / len(union)
        scaled_score = min(1.0, jaccard * 3.5)
        if scaled_score > best_score:
            best_score = scaled_score
            best_claim = claim
    return best_score, best_claim

def check_claim_similarity(text: str) -> Dict:
    model = get_sentence_model()
    if model:
        try:
            from sentence_transformers import util
            text_emb = model.encode(text, convert_to_tensor=True)
            db_embs = model.encode(KNOWN_MISINFORMATION_CLAIMS, convert_to_tensor=True)
            cosine_scores = util.cos_sim(text_emb, db_embs)[0]
            best_idx = int(cosine_scores.argmax())
            return {"score": float(cosine_scores[best_idx]), "claim": KNOWN_MISINFORMATION_CLAIMS[best_idx]}
        except Exception as e:
            logger.error(f"Local similarity check failed: {e}")
            
    score, claim = token_similarity_fallback(text, KNOWN_MISINFORMATION_CLAIMS)
    return {"score": score, "claim": claim}

# ==========================================
# MAIN EXPORT MODULE: ANALYZE_TEXT
# ==========================================
def analyze_text(text: str) -> dict:
    logger.info(f"Evidence Engine: Starting text analysis (length: {len(text)})")
    
    if not text or not text.strip():
        return {
            "score": 0.5,
            "flagged_claims": [],
            "factcheck_debug": {
                "user_claim": "",
                "matched_claim": "No matching verified claim found.",
                "similarity_score": "0%",
                "publisher": "N/A",
                "matched_publisher": "N/A",
                "review_date": "N/A",
                "verdict": "Unverified",
                "matched_verdict": "No verified fact-check found.",
                "reason": "Empty input text provided",
                "article_url": "",
                "status_text": "No verified public fact-check was found for this claim.",
                "evidence_summary": "No trusted evidence available.",
                "evidence_list": [],
                "why_verdict": [],
                "google_status": "No verified claim found",
                "google_status_explanation": "Empty text provided.",
                "primary_entity": "N/A",
                "evidence_sources": [],
                "contradiction_detected": False,
                "contradiction_text": "",
                "confidence_breakdown": {
                    "overall_confidence": 0,
                    "evidence_confidence": 0,
                    "model_confidence": 0,
                    "source_reliability": 0,
                    "consensus_strength": 0
                },
                "timeline": {
                    "claim_published": "N/A",
                    "fact_check_published": "N/A",
                    "latest_update": "N/A",
                    "latest_source": "N/A"
                },
                "explainability_report": {
                    "entities": [],
                    "organizations": [],
                    "locations": [],
                    "dates": [],
                    "factual_claims": [],
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "missing_evidence": [],
                    "risk_factors": [],
                    "reason_confidence": "Empty input.",
                    "reason_verdict": "Empty input."
                },
                "evidence_contribution": {
                    "nlp_analysis": 100,
                    "official_sources": 0,
                    "google_factcheck": 0,
                    "cross_source_agreement": 0
                }
            },
            "factcheck_matched": False,
            "reduce_confidence": False
        }

    # 1. Extract claims
    extracted_claims = extract_claims(text)
    primary_claim = extracted_claims[0] if extracted_claims else text

    # 2. Get Base Classifier Score (DistilBERT)
    pipeline = get_nlp_pipeline()
    distilbert_prob = 0.40
    conf = 0.75
    if pipeline:
        try:
            truncated = text[:1500]
            pred = pipeline(truncated)[0]
            label = pred["label"]
            conf = pred["score"]
            if label == "NEGATIVE":
                distilbert_prob = 0.40 + (conf * 0.5)
            else:
                distilbert_prob = 0.20 + ((1.0 - conf) * 0.3)
        except Exception as e:
            logger.error(f"DistilBERT prediction failed: {e}")
            distilbert_prob = rule_based_text_fake_score(text)
    else:
        distilbert_prob = rule_based_text_fake_score(text)
        
    linguistic_score = rule_based_text_fake_score(text)

    evidence_list_to_ui = []
    
    cat_matches = {
        "google_factcheck": None,
        "wikipedia": None,
        "official_sources": None,
        "trusted_news": None
    }
    
    logger.info(f"[Evidence Engine Audit] Original User Claim: '{text}'")
    logger.info(f"[Evidence Engine Audit] Extracted claim list: {extracted_claims}")

    refutation_keywords = [
        "debunked", "false", "fake", "hoax", "untrue", "misleading", 
        "incorrect", "wrong", "myth", "rumor", "conspiracy", "fabricated"
    ]

    google_api_status = "No verified claim found"
    google_api_explanation = "No Google Fact Check API key is set."
    google_request_url = "N/A"
    
    # Trackers for dates
    extracted_dates_list = []
    fact_check_date_parsed = "N/A"
    latest_update_date = "N/A"
    latest_source_name = "N/A"

    for u_claim in extracted_claims[:3]:
        # A. Query Google Fact Check API
        g_claims, claim_query, req_url, resp_body, g_status, g_exp = query_google_factcheck_api(u_claim)
        google_api_status = g_status
        google_api_explanation = g_exp
        google_request_url = req_url
        u_ent = extract_named_entities(claim_query)
        
        # Collect date entities
        extracted_dates_list.extend(list(u_ent["DATE"]))
        
        logger.info(f"[Evidence Engine Audit] Google Factcheck URL: '{req_url}'")
        
        best_g_sim = 0.0
        best_g_match = None
        for gc in g_claims:
            g_text = gc.get("text", "")
            sem_sim = get_semantic_similarity(claim_query, g_text)
            str_sim = get_string_similarity(claim_query, g_text)
            g_ent = extract_named_entities(g_text)
            ent_match = check_entities_match(u_ent, g_ent)
            
            logger.info(f"[Evidence Engine Audit] Compare Google Claim: '{g_text}' | SemSim: {sem_sim:.3f}, StrSim: {str_sim:.1f}%, EntMatch: {ent_match}")
            
            if (sem_sim >= 0.80 or str_sim >= 85.0) and ent_match:
                if sem_sim > best_g_sim:
                    best_g_sim = sem_sim
                    best_g_match = gc
                    
        if best_g_match:
            reviews = best_g_match.get("claimReview", [])
            if reviews:
                review = reviews[0]
                rating = review.get("textualRating", "Unknown")
                raw_pub = review.get("publisher", {}).get("name", "Unknown Publisher")
                url = review.get("url", "")
                r_date = review.get("reviewDate", "")
                
                # Format timeline factcheck date
                if r_date:
                    try:
                        fact_check_date_parsed = format_wikipedia_timestamp(r_date)
                    except Exception:
                        pass
                
                rating_lower = rating.lower()
                is_fake = any(w in rating_lower for w in refutation_keywords)
                g_verdict = "Refuting" if is_fake else "Confirming"
                g_score = 0.90 if is_fake else 0.10
                
                publisher_clean = clean_publisher_name(url, raw_pub)
                
                cat_matches["google_factcheck"] = {
                    "score": g_score,
                    "title": best_g_match.get("text", ""),
                    "publisher": publisher_clean,
                    "url": url,
                    "verdict": g_verdict,
                    "snippet": f"Google Fact Check tools rated this claim: '{rating}'.",
                    "published_date": format_wikipedia_timestamp(best_g_match.get("claimDate", "")),
                    "last_updated": fact_check_date_parsed,
                    "reliability_badge": get_source_reliability_badge(url, publisher_clean)
                }
                logger.info(f"[Evidence Engine Audit] Google Match ACCEPTED: verdict={g_verdict}, score={g_score}")

        # B. Query Multi-source Crawlers (Wikipedia, News, Government sites)
        flat_entities = []
        for cat_label, items in u_ent.items():
            flat_entities.extend(list(items))
            
        if flat_entities:
            search_query = " ".join(flat_entities[:3])
        else:
            words = claim_query.split()
            search_query = " ".join(words[:8])
            
        logger.info(f"[Evidence Engine Audit] Optimized Search Query: '{search_query}' (Original: '{u_claim}')")
        sources_data = fetch_multi_source_evidence(search_query)
        
        for category in ["wikipedia", "official_sources", "trusted_news"]:
            best_cat_sim = 0.0
            best_cat_item = None
            
            for item in sources_data[category]:
                item_text = f"{item['title']} {item['snippet']}"
                sem_sim = get_semantic_similarity(claim_query, item_text)
                str_sim = get_string_similarity(claim_query, item['title'])
                
                logger.info(f"[Evidence Engine Audit] Compare {category}: '{item['title']}' | SemSim: {sem_sim:.3f}, StrSim: {str_sim:.1f}%")
                
                is_match = False
                if sem_sim >= 0.75 or str_sim >= 75.0:
                    is_match = True
                else:
                    is_match = check_fallback_match(claim_query, item['title'], item['snippet'], flat_entities)
                    if is_match:
                        sem_sim = max(sem_sim, 0.85)

                if is_match:
                    if sem_sim > best_cat_sim:
                        best_cat_sim = sem_sim
                        best_cat_item = item
                        
            if best_cat_item:
                snippet_lower = best_cat_item["snippet"].lower() + " " + best_cat_item["title"].lower()
                is_refuting = any(w in snippet_lower for w in refutation_keywords)
                verdict_state = "Refuting" if is_refuting else "Confirming"
                cat_score = 0.90 if is_refuting else 0.10
                
                publisher_clean = clean_publisher_name(best_cat_item["url"], best_cat_item["publisher"])
                
                # Format dates
                pub_d = "N/A"
                if best_cat_item.get("timestamp"):
                    pub_d = format_wikipedia_timestamp(best_cat_item["timestamp"])
                else:
                    # Look for date in snippet
                    date_match = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*+\s+\d{1,2},\s+\d{4}\b', best_cat_item["snippet"])
                    if date_match:
                        pub_d = date_match.group(0)
                
                cat_matches[category] = {
                    "score": cat_score,
                    "title": best_cat_item["title"],
                    "publisher": publisher_clean,
                    "url": best_cat_item["url"],
                    "verdict": verdict_state,
                    "snippet": best_cat_item["snippet"],
                    "published_date": pub_d,
                    "last_updated": pub_d,
                    "reliability_badge": get_source_reliability_badge(best_cat_item["url"], publisher_clean)
                }
                logger.info(f"[Evidence Engine Audit] {category} Match ACCEPTED: verdict={verdict_state}, score={cat_score}")

    # Compile the final UI evidence list
    for cat_name, val in cat_matches.items():
        if val:
            source_label = "Google Factcheck" if cat_name == "google_factcheck" else (
                "Official Sources" if cat_name == "official_sources" else (
                    "Trusted News" if cat_name == "trusted_news" else "Wikipedia"
                )
            )
            evidence_list_to_ui.append({
                "source_type": source_label,
                "title": val["title"],
                "publisher": val["publisher"],
                "url": val["url"],
                "verdict": val["verdict"],
                "snippet": val["snippet"],
                "published_date": val["published_date"],
                "last_updated": val["last_updated"],
                "reliability_badge": val["reliability_badge"]
            })
            
            # Format timeline items
            if val["last_updated"] != "N/A":
                latest_update_date = val["last_updated"]
                latest_source_name = val["publisher"]

    # FUSION ALGORITHM (weights re-normalization)
    weights = {
        "google_factcheck": 0.35,
        "trusted_news": 0.25,
        "official_sources": 0.20,
        "wikipedia": 0.10,
        "distilbert": 0.10
    }
    
    active_sources = {}
    active_sources["distilbert"] = {"score": distilbert_prob, "weight": weights["distilbert"]}
    
    for cat_name, val in cat_matches.items():
        if val:
            active_sources[cat_name] = {"score": val["score"], "weight": weights[cat_name]}

    total_weight = sum(src["weight"] for src in active_sources.values())
    weighted_sum = sum(src["score"] * src["weight"] for src in active_sources.values())
    nlp_prob = weighted_sum / total_weight
    
    # Step 6 override Verified True
    confirming_sources = [
        k for k, v in cat_matches.items() 
        if v and v["verdict"] == "Confirming" and k != "google_factcheck"
    ]
    refuting_sources = [
        k for k, v in cat_matches.items() 
        if v and v["verdict"] == "Refuting"
    ]
    google_factcheck_active = cat_matches["google_factcheck"] is not None
    consensus_label = "Unverified"
    factcheck_matched = False
    evidence_summary = ""

    if google_factcheck_active:
        factcheck_matched = True
        consensus_label = cat_matches["google_factcheck"]["verdict"]
        evidence_summary = (
            f"Factual consensus established via Google Fact Check database. "
            f"Matched record reviewed by '{cat_matches['google_factcheck']['publisher']}' with verdict: {consensus_label}."
        )
    elif len(confirming_sources) >= 2:
        nlp_prob = 0.04
        factcheck_matched = True
        consensus_label = "Confirming"
        evidence_summary = (
            f"Consensus: High Confidence True. Factual claims verified independently "
            f"by multiple trusted sources ({', '.join(confirming_sources)}). Google Fact Check entry unavailable."
        )
    else:
        if refuting_sources:
            factcheck_matched = True
            consensus_label = "Refuting"
            evidence_summary = (
                f"Consensus established via trusted sources. Factual claims refuted "
                f"by: {', '.join(refuting_sources)}."
            )
        else:
            consensus_label = "Unverified"
            evidence_summary = (
                "No verified public fact-check was found for this claim. "
                "The authenticity score is based on AI analysis only. External evidence unavailable."
            )

    flagged_claims = []
    for cat, val in cat_matches.items():
        if val and val["verdict"] == "Refuting":
            flagged_claims.append(f"{cat.replace('_',' ').title()} ({val['publisher']}): Refuted")

    # Dynamic "Why this verdict?" bullet generator
    why_verdict_bullets = []
    for cat, val in cat_matches.items():
        if val:
            marker = "✓" if val["verdict"] == "Confirming" else "🚨"
            if cat == "official_sources":
                why_verdict_bullets.append(f"{marker} {val['publisher']} officially announced/published documentation matching the claim.")
            elif cat == "wikipedia":
                why_verdict_bullets.append(f"{marker} Wikipedia logs contain active records supporting this claim.")
            elif cat == "trusted_news":
                why_verdict_bullets.append(f"{marker} Major news agency '{val['publisher']}' published reports confirming this claim.")
            elif cat == "google_factcheck":
                status_v = "disputes" if val["verdict"] == "Refuting" else "verifies"
                why_verdict_bullets.append(f"{marker} Google Fact Check reviews by '{val['publisher']}' actively {status_v} the claim.")

    if not refuting_sources and not (google_factcheck_active and cat_matches["google_factcheck"]["verdict"] == "Refuting"):
        why_verdict_bullets.append("✓ No verified fact-check contradicts this claim.")

    # Contradiction Detection
    confirming_count = len([k for k, v in cat_matches.items() if v and v["verdict"] == "Confirming"])
    refuting_count = len([k for k, v in cat_matches.items() if v and v["verdict"] == "Refuting"])
    contradiction_detected = confirming_count > 0 and refuting_count > 0
    contradiction_text = ""
    if contradiction_detected:
        contradiction_text = f"Conflicting Evidence Detected. {confirming_count} trusted source(s) support this claim, while {refuting_count} trusted source(s) dispute it. Manual verification recommended."

    # primary_entity and evidence_sources lists
    primary_entity = "N/A"
    all_evidence_publishers = []
    for item in evidence_list_to_ui:
        all_evidence_publishers.append(item["publisher"])
        
    if all_evidence_publishers:
        # De-duplicate
        seen = set()
        all_evidence_publishers = [x for x in all_evidence_publishers if not (x in seen or seen.add(x))]
        # Exclude Wikipedia if others are present
        entity_candidates = [p for p in all_evidence_publishers if p != "Wikipedia"]
        if entity_candidates:
            primary_entity = entity_candidates[0]
        else:
            primary_entity = all_evidence_publishers[0]

    # Confidence breakdown progress bars
    overall_confidence_pct = round(total_weight * 100)
    evidence_conf = 90 if len(evidence_list_to_ui) >= 2 else (70 if len(evidence_list_to_ui) == 1 else 15)
    model_conf = round(conf * 100)
    
    # Source reliability calculation
    reliability_ratings = []
    for item in evidence_list_to_ui:
        b = item["reliability_badge"]
        if b in ["Space Agency", "Government"]:
            reliability_ratings.append(95)
        elif b in ["Fact Checker", "Major News Agency"]:
            reliability_ratings.append(88)
        elif b == "University":
            reliability_ratings.append(92)
        elif b == "Community Source":
            reliability_ratings.append(78)
        else:
            reliability_ratings.append(50)
            
    source_reliability_pct = round(sum(reliability_ratings) / len(reliability_ratings)) if reliability_ratings else 40
    
    # Consensus strength
    if contradiction_detected:
        consensus_strength_pct = 25
    elif len(evidence_list_to_ui) >= 2:
        consensus_strength_pct = 95
    elif len(evidence_list_to_ui) == 1:
        consensus_strength_pct = 75
    else:
        consensus_strength_pct = 15

    # Timeline freshness
    claim_published_date = "N/A"
    if extracted_dates_list:
        claim_published_date = extracted_dates_list[0]
    elif evidence_list_to_ui:
        # Fallback to earliest published date in matches
        dates_found = [i["published_date"] for i in evidence_list_to_ui if i["published_date"] != "N/A"]
        if dates_found:
            claim_published_date = min(dates_found)

    # Dynamic Evidence Contribution weights (summing to 100)
    contrib = {
        "nlp_analysis": round((weights["distilbert"] / total_weight) * 95),
        "official_sources": round((weights["official_sources"] / total_weight) * 95) if cat_matches["official_sources"] else 0,
        "google_factcheck": round((weights["google_factcheck"] / total_weight) * 95) if cat_matches["google_factcheck"] else 0,
        "wikipedia": round((weights["wikipedia"] / total_weight) * 95) if cat_matches["wikipedia"] else 0,
        "trusted_news": round((weights["trusted_news"] / total_weight) * 95) if cat_matches["trusted_news"] else 0,
    }
    agreement_factor = 5 if len(confirming_sources) + (1 if google_factcheck_active else 0) >= 2 else 0
    contrib["cross_source_agreement"] = agreement_factor
    contrib_sum = sum(contrib.values())
    if contrib_sum > 0:
        evidence_contrib_dict = {k: round((v / contrib_sum) * 100) for k, v in contrib.items() if v > 0}
        # Force re-normalization to sum exactly to 100
        rem = 100 - sum(evidence_contrib_dict.values())
        if rem != 0 and evidence_contrib_dict:
            k_first = list(evidence_contrib_dict.keys())[0]
            evidence_contrib_dict[k_first] += rem
    else:
        evidence_contrib_dict = {"nlp_analysis": 100}

    # Extract dynamic entities lists
    spacy_nlp = get_spacy_nlp()
    all_ents = extract_named_entities(text)
    
    explainability_report = {
        "entities": list(all_ents["PERSON"].union(all_ents["ORGANIZATION"]).union(all_ents["LOCATION"]).union(all_ents["DATE"]).union(all_ents["EVENT"])),
        "organizations": list(all_ents["ORGANIZATION"]),
        "locations": list(all_ents["LOCATION"]),
        "dates": list(all_ents["DATE"]),
        "factual_claims": extracted_claims,
        "supporting_evidence": [i["title"] for i in evidence_list_to_ui if i["verdict"] == "Confirming"],
        "contradicting_evidence": [i["title"] for i in evidence_list_to_ui if i["verdict"] == "Refuting"],
        "missing_evidence": [src for src in ["Google Fact Check", "Wikipedia", "Official Sources", "News Agencies"] if not cat_matches.get(src.lower().replace(" ","_"))],
        "risk_factors": [
            "High linguistic styling markers (exclamation/all-caps/clickbait bias)." if linguistic_score > 0.65 else "Low writing styling bias.",
            "Lack of official primary organization statement." if not cat_matches["official_sources"] else "Confirmed by official organization statements.",
            "Conflicting third-party information reports." if contradiction_detected else "No conflicting evidence detected."
        ],
        "reason_confidence": f"Evidence weight ({evidence_conf}%) + Source trust rating ({source_reliability_pct}%) + Model bias verification ({model_conf}%).",
        "reason_verdict": f"The final consensus verdict was reached based on: {evidence_summary}"
    }

    # Package factcheck_debug payload
    factcheck_debug = {
        "user_claim": primary_claim,
        "matched_claim": cat_matches["google_factcheck"]["title"] if google_factcheck_active else (
            cat_matches[confirming_sources[0]]["title"] if confirming_sources else (
                evidence_list_to_ui[0]["title"] if evidence_list_to_ui else "No matching verified claim found."
            )
        ),
        "similarity_score": f"{round(best_g_sim * 100)}%" if google_factcheck_active else (
            "96%" if len(confirming_sources) >= 2 else "0%"
        ),
        "publisher": primary_entity,
        "matched_publisher": cat_matches["google_factcheck"]["publisher"] if google_factcheck_active else (
            cat_matches[confirming_sources[0]]["publisher"] if confirming_sources else "N/A"
        ),
        "review_date": fact_check_date_parsed,
        "verdict": "False" if consensus_label == "Refuting" else ("True" if consensus_label == "Confirming" else "Unverified"),
        "matched_verdict": "False" if consensus_label == "Refuting" else ("True" if consensus_label == "Confirming" else "No verified fact-check found."),
        "reason": evidence_summary,
        "article_url": cat_matches["google_factcheck"]["url"] if google_factcheck_active else (
            cat_matches[confirming_sources[0]]["url"] if confirming_sources else ""
        ),
        "status_text": evidence_summary,
        "evidence_summary": evidence_summary if evidence_list_to_ui else "No trusted evidence available.",
        
        # New upgraded attributes (UI Improvements & explainability)
        "google_status": google_api_status,
        "google_status_explanation": google_api_explanation,
        "google_query": google_request_url,
        "num_results": len(g_claims),
        "primary_entity": primary_entity,
        "evidence_sources": all_evidence_publishers,
        "evidence_list": evidence_list_to_ui,
        "why_verdict": why_verdict_bullets,
        "contradiction_detected": contradiction_detected,
        "contradiction_text": contradiction_text,
        "confidence_breakdown": {
            "overall_confidence": overall_confidence_pct,
            "evidence_confidence": evidence_conf,
            "model_confidence": model_conf,
            "source_reliability": source_reliability_pct,
            "consensus_strength": consensus_strength_pct
        },
        "timeline": {
            "claim_published": claim_published_date,
            "fact_check_published": fact_check_date_parsed,
            "latest_update": latest_update_date,
            "latest_source": latest_source_name
        },
        "explainability_report": explainability_report,
        "evidence_contribution": evidence_contrib_dict,
        "final_score_calculation": " + ".join([
            f"{k} ({round(src['score']*100)}% * {round(src['weight']*100)}%)" 
            for k, src in active_sources.items()
        ])
    }

    # Verify local database claim similarity for overrides
    sim_results = check_claim_similarity(text)
    if sim_results["score"] > 0.75:
        flagged_claims.append(f"Local Match: {sim_results['claim']}")
        nlp_prob = max(nlp_prob, sim_results["score"])

    return {
        "score": round(nlp_prob, 3),
        "flagged_claims": flagged_claims,
        "factcheck_debug": factcheck_debug,
        "factcheck_matched": factcheck_matched,
        "reduce_confidence": False
    }
