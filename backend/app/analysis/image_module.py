import os
import tempfile
import uuid
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from app.core.config import settings
from app.core.logging import logger

# Lazy PyTorch imports to prevent memory crashes on startup
_cnn_model = None

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    
    class ForensicCNN(nn.Module):
        """
        Lightweight CNN to classify spliced/copy-moved image forensics.
        """
        def __init__(self):
            super(ForensicCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 112x112
                
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 56x56
                
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # 28x28
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 28 * 28, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
            
except ImportError:
    # Set fallback if torch is unavailable during installation
    torch = None
    nn = None
    transforms = None

def get_cnn_model():
    """
    Lazily loads the pretrained CNN model for image manipulation detection.
    """
    global _cnn_model
    if torch is None:
        return False
        
    if _cnn_model is None:
        try:
            model = ForensicCNN()
            weights_path = os.path.join(settings.TORCH_HOME, "image_forensics_casia.pth")
            
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                model.eval()
                _cnn_model = model
                logger.info("Image Forensics: Loaded pretrained CNN weights.")
            else:
                logger.warning(
                    f"Image Forensics: Pretrained weights not found at {weights_path}. "
                    "Inference will fall back to ELA-based scoring."
                )
                _cnn_model = False  # Sentinel
        except Exception as e:
            logger.error(f"Failed to load Image Forensics CNN model: {e}")
            _cnn_model = False
            
    return _cnn_model

def compute_ela(image_path: str, quality: int = 95) -> Tuple[float, int]:
    """
    Error Level Analysis (ELA):
    Saves the image at 95% JPEG quality, computes the absolute difference,
    and analyzes variance. Spliced areas show significantly higher brightness/variance
    because they have different compression histories.
    """
    try:
        # Load original image
        original = Image.open(image_path).convert('RGB')
        
        # Save transient image at quality quality
        temp_dir = tempfile.gettempdir()
        temp_ela_path = os.path.join(temp_dir, f"ela_{uuid.uuid4()}.jpg")
        original.save(temp_ela_path, 'JPEG', quality=quality)
        
        # Load compressed image
        compressed = Image.open(temp_ela_path)
        
        # Convert to numpy arrays
        orig_arr = np.array(original, dtype=np.float32)
        comp_arr = np.array(compressed, dtype=np.float32)
        
        # Compute absolute difference
        diff = np.abs(orig_arr - comp_arr)
        
        # Statistics
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        # Find pixels with high difference (indicating high modification/splicing)
        # Threshold: if average color difference is greater than 15.0
        channel_mean_diff = np.mean(diff, axis=2)
        anomalous_pixels = np.sum(channel_mean_diff > 12.0)
        
        # Group anomalous pixels into "regions" (e.g. 500 pixels form a region)
        manipulated_regions = int(anomalous_pixels // 500)
        
        # Clean up temp file
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)
            
        logger.info(
            f"ELA: Mean Difference: {mean_diff:.4f}, Max Diff: {max_diff}, "
            f"Anomalous Pixels: {anomalous_pixels} ({manipulated_regions} regions)"
        )
        return float(mean_diff), manipulated_regions
        
    except Exception as e:
        logger.error(f"Error computing ELA for {image_path}: {e}")
        return 0.0, 0

def analyze_image(image_path: str) -> dict:
    """
    Image Analysis: Computes Error Level Analysis (ELA) and applies the CNN classifier.
    """
    logger.info(f"Image Forensics Module: analyzing {image_path}")
    
    # 1. Compute ELA
    mean_diff, manipulated_regions = compute_ela(image_path)
    
    # Normalize ELA mean difference to a probability:
    # A mean difference > 4.0 is typical for compression mismatches.
    # We map ELA score between 0.1 and 0.8
    ela_score = min(0.85, 0.1 + (mean_diff / 8.0))
    if manipulated_regions > 5:
        # Boost score if multiple suspicious regions exist
        ela_score = min(0.95, ela_score + 0.15)
        
    # 2. CNN Forensics Classifier
    model = get_cnn_model()
    cnn_score = None
    
    if model and torch is not None:
        try:
            # Prepare image tensor
            img = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                output = model(img_tensor)
                cnn_score = float(torch.sigmoid(output).item())
                
            logger.info(f"Image Forensics: CNN score: {cnn_score:.4f}")
        except Exception as e:
            logger.error(f"Image Forensics: CNN inference failed: {e}")
            
    # Combine scores: if CNN is available, use average. Otherwise fall back to ELA.
    if cnn_score is not None:
        final_score = (0.4 * ela_score) + (0.6 * cnn_score)
    else:
        # ELA fallback score
        final_score = ela_score
        
    return {
        "score": round(final_score, 3),
        "manipulated_regions": manipulated_regions
    }
