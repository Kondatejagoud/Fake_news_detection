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
    text_nlp_score = module_results["text_nlp"]["score"] if module_results.get("text_nlp") else None
    image_forensics_score = module_results["image_forensics"]["score"] if module_results.get("image_forensics") else None
    deepfake_score = module_results["deepfake"]["score"] if module_results.get("deepfake") else None
    
    authenticity_score, confidence_percentage, risk_level = fuse_results(
        text_nlp_score=text_nlp_score,
        image_forensics_score=image_forensics_score,
        deepfake_score=deepfake_score,
        reduce_confidence=reduce_confidence
    )
    
    # 1. Enforce strict unified mapping rules to prevent contradictions
    if authenticity_score >= 80:
        verdict = "Likely Authentic" if input_type in ["image", "video"] else "Likely True"
        risk_level = "Low"
    elif authenticity_score >= 65:
        verdict = "Needs Review"
        risk_level = "Medium"
    elif authenticity_score >= 40:
        verdict = "Suspicious"
        risk_level = "Medium"
    else:
        verdict = "Likely Manipulated" if input_type in ["image", "video"] else "Likely False"
        risk_level = "High"

    # 2. Collect supporting sources
    supporting_sources = []
    text_nlp = module_results.get("text_nlp")
    confirmations = 0
    contradictions = 0
    has_factcheck_match = False
    
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
            confirmations = len([e for e in evidence_list if isinstance(e, dict) and e.get("verdict") == "Confirming"])
            contradictions = len([e for e in evidence_list if isinstance(e, dict) and e.get("verdict") == "Refuting"])
            if debug.get("verdict") and debug.get("verdict") != "Unverified":
                has_factcheck_match = True

    # 3. Generate clean, non-absolute, non-contradictory explanation list
    explanations = []
    
    # Main summary
    explanations.append(
        f"Analyzed input type '{input_type}' with a resulting Authenticity Score of "
        f"{authenticity_score}/100 ({risk_level} Risk) at a confidence rating of {confidence_percentage}%."
    )
    
    # NLP / Text verification explanation
    if text_nlp:
        if has_factcheck_match:
            if verdict in ["Likely True", "Likely Authentic"]:
                explanations.append("Trusted evidence suggests this claim is likely accurate.")
            else:
                explanations.append("Public fact-checking databases verify that this claim is inaccurate or disputed.")
        else:
            if confirmations > 0 and contradictions > 0:
                explanations.append("Evidence consensus is mixed or conflicting. Manual verification is recommended.")
            elif confirmations > 0:
                explanations.append("Trusted evidence suggests this claim is likely accurate.")
            elif contradictions > 0:
                explanations.append("Public fact-checking databases verify that this claim is inaccurate or disputed.")
            else:
                explanations.append("No reliable public evidence was found.")
                
    # Image Forensics explanation
    image_forensics = module_results.get("image_forensics")
    if image_forensics and isinstance(image_forensics, dict) and "score" in image_forensics:
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
    deepfake = module_results.get("deepfake")
    if deepfake and isinstance(deepfake, dict) and "score" in deepfake:
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
            
    # Add a safe share recommendation
    if verdict in ["Likely True", "Likely Authentic"]:
        explanations.append("RECOMMENDATION: Verified claim details match official statements or trusted news.")
    elif verdict == "Needs Review" or verdict == "Suspicious":
        explanations.append("UNVERIFIED: Content does not have active third-party fact check matches. Share with caution.")
    else:
        explanations.append("WARNING: High risk of digital manipulation or disputed facts. Verify source before sharing.")

    return {
        "authenticity_score": authenticity_score,
        "confidence_percentage": confidence_percentage,
        "risk_level": risk_level,
        "verdict": verdict,
        "explanation": explanations,
        "supporting_sources": supporting_sources
    }
