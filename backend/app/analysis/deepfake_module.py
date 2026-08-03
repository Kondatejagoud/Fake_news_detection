import os
import cv2
import numpy as np
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger

_deepfake_model = None

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    
    class DeepfakeClassifier(nn.Module):
        """
        EfficientNet-B0 based binary classifier for deepfake detection.
        Classifies face crops as Real (0) or Fake (1).
        """
        def __init__(self):
            super(DeepfakeClassifier, self).__init__()
            # Use lightweight mobilenet or efficientnet as backbone
            self.backbone = models.mobilenet_v3_small(pretrained=False)
            # Replace classifier with a binary output
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(in_features, 1)

        def forward(self, x):
            return self.backbone(x)
            
except ImportError:
    torch = None
    nn = None
    transforms = None
    models = None

def get_deepfake_model():
    """
    Lazily loads the deepfake classification model.
    """
    global _deepfake_model
    if torch is None:
        return False
        
    if _deepfake_model is None:
        try:
            model = DeepfakeClassifier()
            weights_path = os.path.join(settings.TORCH_HOME, "deepfake_mobilenet.pth")
            
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                model.eval()
                _deepfake_model = model
                logger.info("Deepfake Module: Loaded pretrained Mobilenet weights.")
            else:
                logger.warning(
                    f"Deepfake Module: Pretrained weights not found at {weights_path}. "
                    "Inference will fall back to digital face forensics (pixel statistics)."
                )
                _deepfake_model = False  # Sentinel
        except Exception as e:
            logger.error(f"Failed to load Deepfake model: {e}")
            _deepfake_model = False
            
    return _deepfake_model

def analyze_face_pixels(face_path: str) -> float:
    """
    Fallback forensic pixel-level analysis for face crops.
    Analyzes:
      1. Blurriness (using Laplacian variance) - deepfakes often have soft/blurry facial boundaries.
      2. Pixel noise variance - checks for inconsistencies.
    Returns fake probability score (0.0 to 1.0).
    """
    try:
        img = cv2.imread(face_path)
        if img is None:
            return 0.5

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance (Measure of blurriness)
        # Deepfake boundary warpings tend to yield highly blurry blending borders
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # If lap_var is very low (e.g. < 100), the face image is suspicious of being smoothed/synthesized
        blur_score = 0.0
        if lap_var < 80.0:
            blur_score = 0.5  # High probability of synthetic smoothing/blurring
        elif lap_var < 150.0:
            blur_score = 0.25
            
        # 2. Border inconsistency check (Warping artifacts)
        # Analyze gradients along the borders of the crop
        h, w = gray.shape
        border_pad = int(w * 0.1)
        border_pixels = []
        if border_pad > 0:
            left = gray[:, :border_pad].flatten()
            right = gray[:, -border_pad:].flatten()
            border_pixels = np.concatenate([left, right])
            
        border_std = np.std(border_pixels) if len(border_pixels) > 0 else 0.0
        center_pixels = gray[border_pad:-border_pad, border_pad:-border_pad].flatten()
        center_std = np.std(center_pixels) if len(center_pixels) > 0 else 0.0
        
        # Check standard deviation ratio (unnatural blending boundaries vs centers)
        ratio_score = 0.0
        if center_std > 0 and (border_std / center_std) < 0.4:
            ratio_score = 0.35  # Suspiciously smooth boundaries compared to texture

        score = 0.15 + blur_score + ratio_score
        logger.info(f"Face Forensics (Fallback): Laplacian Var: {lap_var:.2f}, Boundary Std Ratio: {(border_std/center_std) if center_std > 0 else 0:.2f}. Fake prob: {score:.2f}")
        return min(0.95, score)
    except Exception as e:
        logger.error(f"Error analyzing face pixels for {face_path}: {e}")
        return 0.5

def analyze_deepfake(face_image_paths: List[str]) -> dict:
    """
    Deepfake video classifier: analyzes face crops extracted from sampled frames.
    """
    logger.info(f"Deepfake Module: analyzing {len(face_image_paths)} face crops")
    
    if not face_image_paths:
        # No faces detected in video - cannot run deepfake module
        return {
            "score": 0.0,
            "faces_detected": 0
        }

    model = get_deepfake_model()
    scores = []
    
    # Process each face crop
    for path in face_image_paths:
        if not os.path.exists(path):
            continue
            
        score = None
        if model and torch is not None:
            try:
                # Load face image
                img = Image.open(path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    score = float(torch.sigmoid(output).item())
                logger.info(f"Deepfake Classifier: Face crop '{os.path.basename(path)}' score: {score:.4f}")
            except Exception as e:
                logger.error(f"Deepfake Classifier inference failed for {path}: {e}")
                
        # If CNN model not loaded or failed, run forensic pixel analyzer
        if score is None:
            score = analyze_face_pixels(path)
            
        scores.append(score)
        
        # Proactively clean up the temporary cropped face file to preserve disk space
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Could not remove face crop temp file {path}: {e}")

    # Aggregate scores across all face crops (use maximum score as indicator of deepfaking)
    # Why max? If even one frame is clearly deepfaked, the video is considered manipulated.
    final_score = max(scores) if scores else 0.5
    
    return {
        "score": round(final_score, 3),
        "faces_detected": len(scores)
    }
