# Facial Expression Recognition (FER-2013)

A deep learning model that classifies facial expressions into 7 emotions using transfer learning with EfficientNetB2, trained on the FER-2013 dataset.

## Problem

FER-2013 is a well-known but challenging benchmark: 48x48 grayscale facial images across 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise). The dataset has two key challenges:

- **Severe class imbalance** — the `disgust` class makes up only ~1.5% of the data
- **Noisy labels** — facial expressions like fear, sad, and neutral are visually similar even to humans, leading to inherent label ambiguity

## Approach

**Architecture:** EfficientNetB2 (ImageNet pretrained) backbone with a custom classification head (GlobalAveragePooling → BatchNorm → Dense(512) → Dropout → Dense(256) → Dropout → Dense(7, softmax))

**Class imbalance:** Computed balanced class weights via scikit-learn and applied them during training, so misclassifying minority classes like `disgust` is penalized more heavily.

**Data augmentation:** Random horizontal flip, rotation, zoom, translation, and contrast adjustments — applied only during training, never validation.

**Three-phase fine-tuning:**
1. **Phase 1** — Backbone fully frozen, train only the classification head
2. **Phase 2** — Unfreeze top 40 layers of the backbone, fine-tune with a lower learning rate
3. **Phase 3** — Unfreeze top 80 layers, continue fine-tuning at an even lower learning rate (BatchNorm layers kept frozen throughout to preserve pretrained statistics)

**Label smoothing** (0.1) applied to soften overconfident predictions given the dataset's label noise.

## Results

**64% validation accuracy** (FER-2013 published baselines range 60-75%)

| Emotion | Precision | Recall | F1-score |
|---|---|---|---|
| Angry | 0.54 | 0.57 | 0.56 |
| Disgust | 0.57 | 0.68 | 0.62 |
| Fear | 0.56 | 0.43 | 0.48 |
| Happy | 0.85 | 0.80 | 0.82 |
| Neutral | 0.55 | 0.64 | 0.59 |
| Sad | 0.53 | 0.50 | 0.52 |
| Surprise | 0.71 | 0.82 | 0.76 |

The model performs strongly on visually distinct expressions (happy, surprise) and reasonably on the minority disgust class despite limited data. As expected, fear/sad/neutral remain the hardest to separate — a known difficulty in FER-2013 due to overlapping facial cues and label ambiguity.

## Real-time inference

`webcam_inference.py` runs the trained model on a live webcam feed using OpenCV's Haar Cascade for face detection.

```bash
pip install -r requirements.txt
python webcam_inference.py
```

Press `q` to quit.

## Project structure

```
facial-expression-recognition/
├── fer_training.ipynb
├── webcam_inference.py
├── requirements.txt
└── README.md
```

## What I'd improve next

- Test-time augmentation (TTA) for a small inference-time accuracy boost
- Ensemble multiple checkpoints
- Explore ArcFace-style losses for the harder-to-separate classes

## Dataset

[FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
