from typing import Dict, List, Optional, Tuple
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
