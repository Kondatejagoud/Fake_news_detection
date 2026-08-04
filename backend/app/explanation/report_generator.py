from typing import Dict, List, Optional
from app.core.logging import logger

def generate_report(
    input_type: str,
    authenticity_score: int,
    confidence_percentage: int,
    risk_level: str,
    module_results: Dict[str, Optional[Dict]]
) -> List[str]:
    """
    Generates a list of plain-language bullet points explaining the analysis results,
    citing evidence from the active modules.
    """
    explanations = []

    # 1. Base Summary
    explanations.append(
        f"Analyzed input type '{input_type}' with a resulting Authenticity Score of "
        f"{authenticity_score}/100 ({risk_level} Risk) at a confidence rating of {confidence_percentage}%."
    )

    # 2. Text Module Explanations & Google Fact Check integration
    text_results = module_results.get("text_nlp")
    has_factcheck_match = False
    factcheck_verdict = None
    factcheck_publisher = None
    
    if text_results:
        score = text_results.get("score", 0.5)
        conf_pct = round(score * 100)
        flagged = text_results.get("flagged_claims", [])
        debug = text_results.get("factcheck_debug", {})
        
        # Renders the consensus evidence summary directly
        evidence_summary = debug.get("evidence_summary", "")
        if evidence_summary:
            if input_type == "video":
                explanations.append(f"Spoken Claims Audit: {evidence_summary}")
            else:
                explanations.append(f"Consensus Audit: {evidence_summary}")
            
        verdict = debug.get("verdict", "Unverified")
        if verdict != "Unverified":
            has_factcheck_match = True
            factcheck_verdict = verdict
            factcheck_publisher = debug.get("publisher", "Consensus Source")

        # Flag local matches
        local_flagged = [f for f in flagged if "Local Match" in f]
        if local_flagged:
            explanations.append(f"Flagged local claim database match: {local_flagged[0]}.")
            
        # Give reasoning for score
        if has_factcheck_match:
            verdict_lower = factcheck_verdict.lower()
            is_fake = any(w in verdict_lower for w in ["false", "fake", "untrue", "misleading", "incorrect", "debunked", "hoax", "distorted", "manipulated", "wrong"])
            if is_fake:
                if input_type == "video":
                    explanations.append(
                        f"VERDICT FALSE: Active fact checkers confirm the spoken statements are inaccurate/false (Fake probability: {conf_pct}%)."
                    )
                else:
                    explanations.append(
                        f"VERDICT FALSE: Active fact checkers confirm this statement is inaccurate/false (Fake probability: {conf_pct}%)."
                    )
            else:
                if input_type == "video":
                    explanations.append(
                        f"VERDICT TRUE: Active fact checkers verify the spoken statements are accurate/true (Fake probability: {conf_pct}%)."
                    )
                else:
                    explanations.append(
                        f"VERDICT TRUE: Active fact checkers verify this statement is accurate/true (Fake probability: {conf_pct}%)."
                    )
        else:
            # Score reasons when no fact-check is found
            if score > 0.65:
                if input_type == "video":
                    explanations.append(
                        f"NLP analysis of spoken transcript flags potential linguistic bias or hyperbole ({conf_pct}% fake probability)."
                    )
                else:
                    explanations.append(
                        f"Text NLP Classifier flags clickbait or sensational writing styles associated with bias ({conf_pct}% fake probability)."
                    )
            elif score <= 0.40:
                if input_type == "video":
                    explanations.append(
                        f"NLP analysis of spoken transcript indicates neutral statements with standard vocabulary structure ({conf_pct}% fake probability)."
                    )
                else:
                    explanations.append(
                        f"Text NLP Classifier reports low lexical bias, standard journalistic capitalization, and neutral styling ({conf_pct}% fake probability)."
                    )
            else:
                if input_type == "video":
                    explanations.append(
                        f"NLP analysis of spoken transcript indicates normal conversational markers ({conf_pct}% fake probability)."
                    )
                else:
                    explanations.append(
                        f"Text NLP Classifier indicates neutral to borderline writing style markers ({conf_pct}% fake probability)."
                    )

    # 3. Image Forensics Explanations
    image_results = module_results.get("image_forensics")
    if image_results and "score" in image_results:
        score = image_results["score"]
        conf_pct = round(score * 100)
        manipulated = image_results.get("manipulated_regions", 0)

        if score > 0.60:
            explanations.append(
                f"Image Forensics detects high compression variations and potential splicing with {conf_pct}% probability."
            )
            if manipulated > 0:
                explanations.append(f"Specifically flagged {manipulated} anomalous region(s) using Error Level Analysis (ELA).")
        else:
            explanations.append(
                "Image Forensics check indicates digital composition stability. No obvious signs of splicing or double-compression detected."
            )

    # 4. Deepfake Explanations
    deepfake_results = module_results.get("deepfake")
    if deepfake_results and "score" in deepfake_results:
        score = deepfake_results["score"]
        conf_pct = round(score * 100)
        faces = deepfake_results.get("faces_detected", 0)

        if score > 0.60:
            explanations.append(
                f"Deepfake Facial Classifier detects synthetic manipulation patterns on {faces} scanned face(s) with {conf_pct}% probability."
            )
        else:
            explanations.append(
                f"Deepfake analysis scanned {faces} face(s) across sampled frames. Face metrics appear natural."
            )

    # 5. Dynamic Verdict-Based Recommendations
    if has_factcheck_match:
        if risk_level == "High" or score >= 0.80:
            explanations.append(
                f"RECOMMENDATION: DO NOT SHARE. Fact Checkers at '{factcheck_publisher}' have rated this statement as '{factcheck_verdict}'."
            )
        else:
            explanations.append(
                f"RECOMMENDATION: Safe to Share. Fact Checkers at '{factcheck_publisher}' have verified this claim as '{factcheck_verdict}'."
            )
    else:
        if risk_level == "High":
            explanations.append("WARNING: High risk of metadata editing/splicing anomalies. Verify source before sharing.")
        elif risk_level == "Medium" or risk_level == "Low":
            explanations.append("UNVERIFIED: Content does not have active third-party fact check matches. Share with caution.")

    return explanations
