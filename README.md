# Deep Feature Extraction + Classical Classifiers on MedMNIST

Medical image classification on four MedMNIST subsets using deep features extracted from pretrained CNN/Transformer backbones, classified with classical machine learning models.

The idea: instead of training a deep network end-to-end, use a pretrained backbone (ViT-B/16 or ResNet-18) as a frozen feature extractor and train a lightweight classifier (SVM / XGBoost) on top. The approach was first developed and tuned on BreastMNIST, then applied to three more subsets for comparison.

## Pipeline

```
MedMNIST images → resize/normalize → pretrained backbone (ViT-B/16 / ResNet-18)
→ deep feature vectors → scaling → SVM / XGBoost → evaluation (Accuracy, F1, AUC)
```

- Backbones: ViT-B/16, ResNet-18 (ImageNet pretrained, frozen)
- Classifiers: SVM (RBF, tuned with GridSearchCV), XGBoost
- Metrics: test accuracy, F1 (macro for multi-class), AUC (macro for multi-class)

## Results (best combination per subset)

| Subset | Classes | Best Model | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| BreastMNIST | 2 | ViT-B/16 + SVM (C=5, gamma=scale) | 88.5% | 0.923 | 0.925 |
| BloodMNIST | 8 | ViT-B/16 + XGBoost | 87.6% | 0.858 | 0.986 |
| DermaMNIST | 7 | ViT-B/16 + XGBoost | 73.4% | 0.554 | 0.914 |
| PathMNIST | 9 | ViT-B/16 + XGBoost | 90.5% | 0.871 | 0.992 |

## Observations

- ViT-B/16 features outperformed ResNet-18 features on all four subsets.
- XGBoost was the best classifier on 3 of 4 subsets.
- DermaMNIST has heavy class imbalance (~67% of samples in one class), which explains the large gap between accuracy and macro F1.
- PathMNIST, with relatively balanced classes, gave the most consistent accuracy/F1 results.
- On BloodMNIST, RBF SVM was trained on a 1500-sample subsample because it doesn't scale well to the full ~18.6k training set.

## Project Structure

```
main_breastmnist.py   # BreastMNIST experiments (first subset, tuning)
main_bloodmnist.py    # BloodMNIST experiments
main_dermamnist.py    # DermaMNIST experiments
main_pathmnist.py     # PathMNIST experiments
requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main_breastmnist.py
```

The datasets are downloaded automatically by the `medmnist` package on first run. A CUDA-capable GPU is recommended for feature extraction (tested on an RTX 4070 Laptop GPU).

## Dataset

MedMNIST v2 — Yang, J. et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.
