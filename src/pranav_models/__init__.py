"""
Pranav's Pattern-Aware Multi-Task Learning System
95.69% accuracy on CIFAKE dataset
"""

from .pattern_aware_cnn import PatternAwareResNet
from .pattern_utils import compute_all_patterns
from .pattern_trainer import train_pattern_aware

__all__ = ['PatternAwareResNet', 'compute_all_patterns', 'train_pattern_aware']
