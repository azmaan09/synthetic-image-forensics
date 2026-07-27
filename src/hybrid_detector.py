"""
Hybrid Detector: Pattern-Aware CNN + Forensic Features
Combines Pranav's CNN (95.69%) with Aman's forensic analysis
Expected: 96-97% accuracy
"""

import torch
import numpy as np
from .pranav_models.pattern_aware_cnn import PatternAwareResNet
from .forensic_features import extract_all_features
import pickle

class HybridEnsemble:
    """
    Ensemble combining:
    - Pattern-aware CNN (learns from pixels)
    - Forensic features (analyzes frequency/noise/statistics)
    
    Voting strategy: Weighted average
    """
    
    def __init__(self, cnn_path, forensic_path, cnn_weight=0.6):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Pranav's pattern-aware CNN
        self.cnn = PatternAwareResNet()
        checkpoint = torch.load(cnn_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint['model_state_dict'])
        self.cnn.to(self.device)
        self.cnn.eval()
        
        # Load Aman's forensic model
        with open(forensic_path, 'rb') as f:
            self.forensic_model = pickle.load(f)
        
        self.cnn_weight = cnn_weight
        self.forensic_weight = 1 - cnn_weight
    
    def predict(self, image_path, transform):
        """
        Hybrid prediction
        
        Returns:
            prediction: 0 (FAKE) or 1 (REAL)
            confidence: float [0, 1]
            explanations: dict with scores from both models
        """
        # CNN prediction
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_logits, pattern_scores = self.cnn(img_tensor)
            cnn_probs = torch.softmax(cnn_logits, dim=1)[0].cpu().numpy()
        
        # Forensic prediction
        features = extract_all_features(image_path)
        forensic_probs = self.forensic_model.predict_proba([features])[0]
        
        # Weighted ensemble
        ensemble_probs = (self.cnn_weight * cnn_probs + 
                         self.forensic_weight * forensic_probs)
        
        prediction = np.argmax(ensemble_probs)
        confidence = ensemble_probs[prediction]
        
        explanations = {
            'cnn_prediction': np.argmax(cnn_probs),
            'cnn_confidence': float(cnn_probs[np.argmax(cnn_probs)]),
            'forensic_prediction': np.argmax(forensic_probs),
            'forensic_confidence': float(forensic_probs[np.argmax(forensic_probs)]),
            'pattern_scores': pattern_scores[0].cpu().numpy().tolist(),
            'ensemble_confidence': float(confidence)
        }
        
        return int(prediction), float(confidence), explanations

# This combines best of both:
# - CNN: Learns complex pixel patterns (95.69%)
# - Forensic: Analyzes statistical signatures
# - Expected: 96-97% accuracy
