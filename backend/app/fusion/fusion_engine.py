from typing import Dict, List, Optional, Tuple, Any
from app.core.logging import logger

# Default reliability weights for each module
# Text is reliable, but image and deepfake (video) models have high weights
# when present due to their specific forensic nature.
DEFAULT_WEIGHTS = {
    "text_nlp": 0.35,
    "image_forensics": 0.40,
    "deepfake": 0.45
}

def calculate_disagreement_penalty(scores: List[float]) -> float:
    """
    Computes a confidence penalty based on how widely the modules disagree.
    If they differ widely (e.g., NLP says real [0.1] but Image says fake [0.9]),
    we reduce confidence.
    """
    if len(scores) <= 1:
        return 0.0
    
    # Maximum difference between any two scores
    max_diff = max(scores) - min(scores)
    
    # Penalty goes up to 30% if max difference is 1.0 (complete disagreement)
    return max_diff * 30.0

def run_meta_classifier(features: Dict[str, float]) -> Optional[float]:
    """
    Future extension point to swap out the weighted average for a trained
    meta-classifier (e.g., Logistic Regression or small Multi-Layer Perceptron).
    
    Currently returns None to fall back to the weighted average.
    
    Args:
        features: dictionary of active module scores (normalized 0-1)
    """
    # TODO: Implement logistic regression or small MLP:
    # 1. Load model weights
    # 2. Vectorize features (with imputation for missing values)
    # 3. Model inference: probability = model.predict_proba(features)
    # 4. Return probability
    return None

def fuse_results(
    text_nlp_score: Optional[float] = None,
    image_forensics_score: Optional[float] = None,
    deepfake_score: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    reduce_confidence: bool = False
) -> Tuple[int, int, str]:
    """
    Combines the active analysis module scores to output:
      1. authenticity_score (0 - 100)
      2. confidence_percentage (0 - 100)
      3. risk_level ('Low', 'Medium', 'High')
    
    Scores are expected to be 0.0 - 1.0 fake probability (1.0 = definitely fake).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    active_scores = {}
    if text_nlp_score is not None:
        active_scores["text_nlp"] = max(0.0, min(1.0, text_nlp_score))
    if image_forensics_score is not None:
        active_scores["image_forensics"] = max(0.0, min(1.0, image_forensics_score))
    if deepfake_score is not None:
        active_scores["deepfake"] = max(0.0, min(1.0, deepfake_score))

    if not active_scores:
        # Defaults if no modules ran
        logger.warning("No active modules detected in fusion. Returning default values.")
        return 50, 50, "Medium"

    # Try running the ML meta-classifier first
    fake_probability = run_meta_classifier(active_scores)
    
    if fake_probability is None:
        # Fallback to the baseline weighted average
        weighted_sum = sum(active_scores[m] * weights[m] for m in active_scores)
        weight_sum = sum(weights[m] for m in active_scores)
        fake_probability = weighted_sum / weight_sum
        logger.info(f"Fusion: Weighted average fake probability computed: {fake_probability:.4f}")
    else:
        logger.info(f"Fusion: ML Meta-classifier fake probability computed: {fake_probability:.4f}")

    # Calculate Authenticity Score: 100 is authentic, 0 is fake
    authenticity_score = round((1.0 - fake_probability) * 100)

    # Determine Risk Level based on authenticity
    if authenticity_score >= 70:
        risk_level = "Low"
    elif authenticity_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Calculate Confidence Percentage
    # Base confidence goes up with more data sources (modules)
    num_modules = len(active_scores)
    if num_modules == 1:
        base_confidence = 75.0
    elif num_modules == 2:
        base_confidence = 88.0
    else:
        base_confidence = 96.0

    penalty = calculate_disagreement_penalty(list(active_scores.values()))
    if reduce_confidence:
        penalty += 20.0
        
    confidence_percentage = max(10, min(100, round(base_confidence - penalty)))
    
    logger.info(
        f"Fusion Result - Authenticity: {authenticity_score}, "
        f"Confidence: {confidence_percentage}%, Risk: {risk_level}"
    )

    return authenticity_score, confidence_percentage, risk_level

def compute_final_decision(
    input_type: str,
    module_results: Dict[str, Optional[Dict]],
    reduce_confidence: bool = False
) -> Dict[str, Any]:
    """
    Centralized Final Decision Engine that combines all active module outputs
    into a single consistent Final Decision Object.
    """
    text_nlp = module_results.get("text_nlp")
    image_forensics = module_results.get("image_forensics")
    deepfake = module_results.get("deepfake")

    text_nlp_score = text_nlp["score"] if text_nlp else None
    image_forensics_score = image_forensics["score"] if image_forensics else None
    deepfake_score = deepfake["score"] if deepfake else None
    
    authenticity_score, confidence_percentage, risk_level = fuse_results(
        text_nlp_score=text_nlp_score,
        image_forensics_score=image_forensics_score,
        deepfake_score=deepfake_score,
        reduce_confidence=reduce_confidence
    )
    
    # 1. Enforce strict unified mapping rules to prevent contradictions (Patch 5)
    if authenticity_score >= 80:
        verdict = "TRUE"
        risk_level = "Low"
        recommendation = "Likely authentic."
    elif authenticity_score >= 65:
        verdict = "NEEDS REVIEW"
        risk_level = "Medium"
        recommendation = "Independent verification recommended."
    elif authenticity_score >= 40:
        verdict = "UNVERIFIED"
        risk_level = "Medium"
        recommendation = "Insufficient evidence."
    else:
        verdict = "FALSE"
        risk_level = "High"
        recommendation = "Likely misinformation."

    # 2. Populate Supporting Evidence and Contradicting Evidence dynamically (Patch 4)
    supporting_evidence = []
    contradicting_evidence = []
    evidence_sources = []
    primary_entity = "Community Source (Wikipedia)"
    named_entities = []

    confirmations = 0
    contradictions_count = 0
    has_factcheck_match = False
    
    if text_nlp and isinstance(text_nlp, dict):
        debug = text_nlp.get("factcheck_debug", {})
        if debug and isinstance(debug, dict):
            primary_entity = debug.get("primary_entity", primary_entity)
            evidence_list = debug.get("evidence_list", [])
            for ev in evidence_list:
                if isinstance(ev, dict):
                    pub = ev.get("publisher", "Unknown Publisher")
                    v_state = ev.get("verdict", "Neutral")
                    url = ev.get("url") or ""
                    title = ev.get("title") or ""
                    snippet = ev.get("snippet") or ""
                    source_type = ev.get("source_type", "Source")
                    reliability_badge = ev.get("reliability_badge", "N/A")
                    reliability_score = ev.get("reliability_score", 50)
                    pub_date = ev.get("published_date", "N/A")
                    last_up = ev.get("last_updated", "N/A")
                    strength = ev.get("evidence_strength", "Neutral")
                    
                    evidence_item = {
                        "publisher": pub,
                        "verdict": v_state,
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "source_type": source_type,
                        "reliability_badge": reliability_badge,
                        "reliability_score": reliability_score,
                        "published_date": pub_date,
                        "last_updated": last_up,
                        "evidence_strength": strength
                    }
                    
                    if v_state == "Confirming":
                        supporting_evidence.append(evidence_item)
                        confirmations += 1
                    elif v_state == "Refuting":
                        contradicting_evidence.append(evidence_item)
                        contradictions_count += 1
                    else:
                        supporting_evidence.append(evidence_item)
                    
                    if pub and pub not in evidence_sources:
                        evidence_sources.append(pub)
            
            if debug.get("verdict") and debug.get("verdict") != "Unverified":
                has_factcheck_match = True
        
        # Build named entities from entities_typed (Name (TYPE))
        entities_typed = text_nlp.get("entities_typed", {})
        if entities_typed:
            type_mapping = {
                "PERSON": "PERSON",
                "ORGANIZATION": "ORGANIZATION",
                "LOCATION": "LOCATION",
                "COUNTRY": "LOCATION",
                "DATE": "DATE",
                "EVENT": "EVENT"
            }
            for t_key, t_label in type_mapping.items():
                names = entities_typed.get(t_key, [])
                for name in names:
                    item_str = f"{name} ({t_label})"
                    if item_str not in named_entities:
                        named_entities.append(item_str)

    # 3. Dynamic Confidence explanations (Patch 7)
    active_scores = []
    if text_nlp_score is not None:
        active_scores.append(text_nlp_score)
    if image_forensics_score is not None:
        active_scores.append(image_forensics_score)
    if deepfake_score is not None:
        active_scores.append(deepfake_score)
        
    num_modules = len(active_scores)
    base_confidence = 75.0 if num_modules == 1 else (88.0 if num_modules == 2 else 96.0)
    penalty = calculate_disagreement_penalty(active_scores)
    
    confidence_reasons = []
    confidence_reasons.append(f"Confidence base rating is {base_confidence}% based on {num_modules} active analysis module(s).")
    if penalty > 0:
        confidence_reasons.append(f"Confidence reduced by {penalty:.1f}% due to disagreement in active module scores.")
    if reduce_confidence:
        confidence_reasons.append("Confidence reduced because only unverified third-party sources were found.")
        
    confidence_reason = " ".join(confidence_reasons)

    # 4. Generate clean, non-absolute, non-contradictory explanation list (Patch 2)
    explanations = []
    
    # Verdict summary bullet
    explanations.append(
        f"Analyzed input type '{input_type}' with a resulting Authenticity Score of "
        f"{authenticity_score}/100 and a consensus confidence of {confidence_percentage}%."
    )
    
    # NLP / Text evidence check (Ensure no contradictions)
    if text_nlp:
        if verdict == "TRUE":
            if supporting_evidence:
                explanations.append("Trusted evidence suggests this claim is likely accurate.")
            else:
                explanations.append("Linguistic analysis indicates a neutral reporting style with low lexical bias.")
        elif verdict == "FALSE":
            if contradicting_evidence:
                explanations.append("Public fact-checking databases verify that this claim is inaccurate or disputed.")
            else:
                explanations.append("Linguistic analysis flags high writing styling bias associated with misinformation.")
        elif verdict == "NEEDS REVIEW":
            if confirmations > 0 and contradictions_count > 0:
                explanations.append("Evidence consensus is mixed or conflicting. Independent verification recommended.")
            else:
                explanations.append("Evidence consensus is incomplete. Manual verification is recommended.")
        else: # UNVERIFIED
            explanations.append("No reliable public evidence was found.")

    # Image Forensics explanation
    if image_forensics and "score" in image_forensics:
        img_score = image_forensics.get("score", 0.0)
        manipulated = image_forensics.get("manipulated_regions", 0)
        if img_score > 0.60:
            explanations.append(
                "Image analysis detects high frequency noise deviations, suggesting potential splicing or localization anomalies."
            )
            if manipulated > 0:
                explanations.append(f"Specifically flagged {manipulated} anomalous region(s) using Error Level Analysis (ELA).")
        else:
            explanations.append(
                "Image analysis reports typical double-compression stability without obvious anomalies."
            )

    # Deepfake detection explanation
    if deepfake and "score" in deepfake:
        df_score = deepfake.get("score", 0.0)
        faces = deepfake.get("faces_detected", 0)
        if df_score > 0.60:
            explanations.append(
                f"Facial analysis flags patterns consistent with synthetic face generation or frame manipulation on {faces} face(s)."
            )
        else:
            explanations.append(
                f"Facial analysis scanned {faces} face(s) across sampled frames. Face metrics appear natural."
            )
            
    # Include recommendation explanation
    explanations.append(f"RECOMMENDATION: {recommendation}")

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "explanation": explanations,
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "confidence": confidence_percentage,
        "evidence_sources": evidence_sources,
        "primary_entity": primary_entity,
        "named_entities": named_entities,
        "confidence_reason": confidence_reason,
        "authenticity_score": authenticity_score,
        "confidence_percentage": confidence_percentage
    }
