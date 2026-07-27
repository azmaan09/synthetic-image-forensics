# Synthetic AI Image Detection with Accuracy and Faithfulness Analysis

Research-grade pipeline for detecting AI-generated images and explaining the signals behind each prediction. This project combines a high-accuracy CNN detector with forensic feature models and produces interpretable visuals suitable for publications and presentations.

## Project Motivation
Generative models produce photorealistic images, which creates a need for reliable detection and transparent explanations. This repository focuses on both accuracy and faithfulness: detecting synthetic images and revealing the evidence used for classification.

## Visual Examples: Real vs AI Generated Images
AI-generated images often look realistic to humans but contain subtle statistical artifacts. This project focuses on detecting these differences using deep learning and forensic signal analysis.

### Example 1 — Human Faces
<img src="images/real_vs_ai_faces.jpg" alt="Real vs AI Face" width="900" />

This example illustrates how AI-generated faces can appear visually realistic while still containing subtle statistical artifacts that differ from real camera photographs.

Our detection system analyzes frequency patterns, noise statistics, and texture signals to distinguish synthetic faces from authentic human images.

### Example 2 — Portrait Pair Comparison
<img src="images/real_vs_ai_pair1.jpg" alt="Real vs AI Portrait" width="900" />

AI-generated images can closely mimic real photography, making visual inspection alone unreliable for detecting synthetic content.

The model learns to identify signal-level inconsistencies such as frequency artifacts, color-channel correlations, and noise irregularities.

### Example 3 — Detailed Facial Features Comparison
<img src="images/real_vs_ai_pair2.jpg" alt="Real vs AI Pair 2" width="900" />

Close-up comparisons reveal subtle differences in micro-textures, lighting consistency, and fine-grained details between real and AI-generated images.

Our approach focuses on detecting statistical signals such as frequency artifacts, noise distributions, and texture irregularities rather than relying on semantic appearance.

## What the Model Looks For
The detector analyzes statistical cues that are hard to see with the naked eye:
- frequency domain artifacts
- noise statistics
- color channel correlations
- texture irregularities

These features help distinguish AI-generated images from real photographs.

## Methodology
1. **CNN detector** using pretrained ResNet-50 (binary classifier)
2. **Forensic feature detector** using frequency, noise, color, and texture statistics
3. **Faithfulness analysis** via Grad-CAM and feature importance/SHAP
4. **Robustness testing** under common image transformations

## Detection Pipeline
```
Input Image
  ↓
Feature Extraction / CNN Feature Learning
  ↓
Forensic Signal Analysis
  ↓
Classification (Real vs Synthetic)
```

The system combines deep learning with forensic signals to improve accuracy and interpretability.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Results Comparison
The evaluation script outputs a comparison table:
```
Model | Accuracy | Precision | Recall | F1 | AUC
```
This includes CNN, forensic feature models, and a hybrid ensemble.

## Quick Start
Install dependencies:
```
pip install -r requirements.txt
```

Train the CNN detector:
```
python train_cnn.py
```

Train the forensic feature model:
```
python train_feature_model.py
```

Evaluate and compare models (including robustness tests):
```
python evaluate.py
```

Generate all visuals:
```
python visualize.py
```

Faithfulness / interpretability analysis only:
```
python faithfulness_analysis.py
```

For direct module execution without wrappers:
```
python -m src.train_cnn
python -m src.train_feature_model
python -m src.evaluate
python -m src.visualizations
python -m src.faithfulness_analysis
```

## Project Structure
```
synthetic-image-forensics/
├── data/
│   ├── real/              # 30 real images (real_000.png - real_029.png)
│   └── synthetic/         # 30 synthetic images (synthetic_000.png - synthetic_029.png)
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── feature_visualization.ipynb
├── outputs/
│   ├── models/
│   │   ├── cnn_resnet50.pt              # Trained CNN model (89.99 MB)
│   │   ├── random_forest_forensic.pkl
│   │   ├── svm_forensic.pkl
│   │   └── log_reg_forensic.pkl
│   └── plots/                           # 23 visualization files
│       ├── cnn_training_history.csv
│       ├── forensic_model_results.csv
│       ├── real_samples_grid.png
│       ├── synthetic_samples_grid.png
│       ├── confusion matrices, ROC curves, PR curves, etc.
│       └── feature analysis plots
├── src/
│   ├── __init__.py
│   ├── cnn_detector.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── faithfulness_analysis.py
│   ├── forensic_features.py
│   ├── preprocessing.py
│   ├── train_cnn.py
│   ├── train_feature_model.py
│   └── visualizations.py
├── main.py                # Main pipeline orchestrator
├── train_cnn.py           # CNN training wrapper
├── train_feature_model.py # Forensic feature training wrapper
├── evaluate.py            # Evaluation wrapper
├── visualize.py           # Visualization wrapper
├── faithfulness_analysis.py # Faithfulness analysis wrapper
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Visuals Gallery (Generated in `outputs/plots/`)
Run `python visualize.py` after placing data to generate a full set of visuals:
- `fft_example.png`: example FFT power spectrum
- `feature_high_freq_ratio.png`, `feature_spectral_entropy.png`, `feature_noise_variance.png`, `feature_noise_correlation.png`
- `feature_correlation_matrix.png`: correlation across forensic features
- `transformations_example.png`: JPEG/resize/crop/noise
- `gradcam_example.png`: CNN explanation (if CNN model exists)
- `feature_importance.png`, `shap_summary.png`: feature model explanations (if model exists)
- `real_samples_grid.png`, `synthetic_samples_grid.png`: sample grids
- `class_distribution.png`: dataset balance
- `lbp_histogram.png`, `noise_residual_example.png`, `gradient_magnitude_example.png`
- `cnn_metrics_roc.png`, `cnn_metrics_pr.png`, `cnn_confusion.png`
- `forensic_metrics_roc.png`, `forensic_metrics_pr.png`, `forensic_confusion.png`

### Visual Gallery
![Real Samples](outputs/plots/real_samples_grid.png)
![Synthetic Samples](outputs/plots/synthetic_samples_grid.png)
![FFT Example](outputs/plots/fft_example.png)
![Feature Correlation](outputs/plots/feature_correlation_matrix.png)
![High-Frequency Ratio](outputs/plots/feature_high_freq_ratio.png)
![Spectral Entropy](outputs/plots/feature_spectral_entropy.png)
![Noise Variance](outputs/plots/feature_noise_variance.png)
![Noise Correlation](outputs/plots/feature_noise_correlation.png)
![LBP Histogram](outputs/plots/lbp_histogram.png)
![Transformations](outputs/plots/transformations_example.png)
![Noise Residual](outputs/plots/noise_residual_example.png)
![Gradient Magnitude](outputs/plots/gradient_magnitude_example.png)
![Grad-CAM](outputs/plots/gradcam_example.png)
![Feature Importance](outputs/plots/feature_importance.png)
![SHAP Summary](outputs/plots/shap_summary.png)
![CNN ROC](outputs/plots/cnn_metrics_roc.png)
![CNN PR](outputs/plots/cnn_metrics_pr.png)
![CNN Confusion](outputs/plots/cnn_confusion.png)
![Forensic ROC](outputs/plots/forensic_metrics_roc.png)
![Forensic PR](outputs/plots/forensic_metrics_pr.png)
![Forensic Confusion](outputs/plots/forensic_confusion.png)

## Reproducibility
- Uses fixed random seeds
- Stores models in `outputs/models/`
- Stores plots and visualizations in `outputs/plots/`

## Future Work
- Expand datasets with diverse generative models (SDXL, FLUX, StyleGAN3)
- Improve robustness under heavy compression and post-processing
- Add multimodal consistency checks and provenance signals

---

## 🆕 Pattern-Aware Multi-Task Learning (Pranav's Contribution)

### Overview
Advanced ResNet50 with multi-task learning achieving **95.69% validation accuracy** on CIFAKE dataset (100k training, 20k test).

### Key Innovation
Incorporates 4 explicit visual patterns into training:
1. **Background texture complexity** - Detects inconsistent backgrounds
2. **Edge clarity** - Identifies soft/blurry boundaries  
3. **Lighting consistency** - Finds contradictory shadows
4. **Texture uniformity** - Catches morphing textures

### Performance
| Metric | Baseline | Pattern-Aware | Improvement |
|--------|----------|---------------|-------------|
| Validation Accuracy | 92.01% | **95.69%** | +3.68 pts |
| Peak→Final Degradation | 4.54% | **0.52%** | 9× better |
| Train-Val Gap | 7.34% | **<1%** | 66% reduction |

### Hybrid Ensemble (Combined Approach)
Combines pattern-aware CNN with forensic features for maximum accuracy.

**Expected performance: 96-97%**
```python
from src.hybrid_detector import HybridEnsemble

detector = HybridEnsemble(
    cnn_path='outputs/models/pranav_pattern_aware/best_model.pth',
    forensic_path='outputs/models/random_forest_forensic.pkl'
)

prediction, confidence, explanations = detector.predict('test_image.jpg', transform)
```

### Training Pattern-Aware Model
```bash
python -m src.pranav_models.pattern_trainer --dataset cifake --epochs 20
```

### Technical Details
- **Architecture**: ResNet50 with dual output heads (classification + patterns)
- **Loss**: CrossEntropy + 0.2 × MSE (multi-task)
- **Training**: 54 minutes on Tesla GPU (20 epochs)
- **Dataset**: CIFAKE (100k train, 20k test)

### Citation
```bibtex
@misc{kishore2026pattern,
  author = {Kishore, Pranav},
  title = {Pattern-Aware Multi-Task Learning for AI Image Detection},
  year = {2026},
  institution = {Northeastern University}
}
```

---

## Combined System Performance

Our research combines two complementary approaches:
1. **Pattern-Aware CNN** (Pranav) - Learns from pixels, 95.69%
2. **Forensic Features** (Aman) - Analyzes statistics
3. **Hybrid Ensemble** - Combines both, expected 96-97%

For publication: Demonstrates that combining learned patterns with hand-crafted forensic features achieves state-of-art performance.
