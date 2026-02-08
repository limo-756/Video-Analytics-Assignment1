
# Comparative Analysis of Video Classification Models

## 1. Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Classical (SVM - Dense)** | 0.9667 (Val) | ~0.97 | ~0.97 | ~0.97 |
| **Classical (RF - Dense)** | 0.9667 (Val) | ~0.97 | ~0.97 | ~0.97 |
| **Classical (KNN - Dense)** | 0.8000 (Val) | ~0.80 | ~0.80 | ~0.80 |
| **Classical (Frame Voting)** | **1.0000 (Val)** | **1.0000** | **1.0000** | **1.0000** |
| **Deep Learning (2D CNN)** | **1.0000 (Test)** | 1.0000 | 1.0000 | 1.0000 |
| **Deep Learning (R(2+1)D)** | 0.9333 (Test) | 0.9444 | 0.9333 | 0.9327 |

*Note: Classical model metrics derived from Validation Set. Deep Learning metrics from Test Set.*
	
### Visual Performance Analysis

#### Confusion Matrices
| Classical (Voting) | Deep Learning (2D CNN) | Deep Learning (R(2+1)D) |
|:---:|:---:|:---:|
| ![Classical CM](results_classical/cm_dense_stdscaler_voting.png) | ![2D CNN CM](results_deep_learning_2dcnn/confusion_matrix.png) | ![R(2+1)D CM](results_deep_learning_r2plus1d/confusion_matrix.png) |

#### ROC Curves
| Classical (SVM) | Deep Learning (2D CNN) | Deep Learning (R(2+1)D) |
|:---:|:---:|:---:|
| ![Classical ROC](results_classical/roc_dense_stdscaler_svm.png) | ![2D CNN ROC](results_deep_learning_2dcnn/roc_curve.png) | ![R(2+1)D ROC](results_deep_learning_r2plus1d/roc_curve.png) |


## 2. Computational Efficiency

| Metric | Classical (Best - Voting) | Deep Learning (2D CNN) | Deep Learning (R(2+1)D) |
|--------|------------------|------------------------|-------------------------|
| **Training Time** | ~1-2 min (excl featurization) / ~5 min (total) | ~61s | ~110s |
| **Inference Time (per video)** | ~0.33s (dominated by feature extraction) | ~0.0005s (GPU inference only) / ~0.05s (total) | ~0.0158s (GPU) |
| **GPU Memory** | N/A (CPU) | ~961 MB | Efficient |
| **Model Size** | ~100s KB (Saved Models) | ~45 MB (11M params) | ~120 MB (31M params) |

## 3. Data Efficiency Analysis
- **Classical Methods**: Generally require careful feature engineering. With limited data, they might struggle if features aren't robust, but can work well with small datasets if handcrafted features are good.
- **Deep Learning (2D CNN)**: Showed extreme efficiency (perfect score) suggesting the pre-trained ResNet18 backbone is highly effective and data-efficient for this specific task (transfer learning).
- **Deep Learning (R(2+1)D)**: Also highly data efficient, leveraging Kinetics-400 pre-training.

## 4. Strengths and Limitations

### Classical Approaches
- **Strengths**: Interpretability (via feature importance), lower hardware requirements for inference (CPU).
- **Limitations**: Requires heavy feature engineering, lacks end-to-end learning capabilities, separation of feature extraction and classification.

### Deep Learning (2D CNN)
- **Strengths**: Extremely fast inference, perfect accuracy on this test set, simple architecture.
- **Limitations**: Ignores temporal dynamics (frame averaging), might fail on complex temporal actions.

### Deep Learning (R(2+1)D)
- **Strengths**: Captures 3D spatiotemporal features, robust to temporal variations.
- **Limitations**: Higher computational cost, larger model size.

## 5. Feature/Representation Analysis
- **Classical**: Features (Color, Texture, Motion/Optical Flow) are manually defined.
- **Deep Learning**:
    - **2D CNN**: t-SNE shows perfect separation based on spatial features.
    - **R(2+1)D**: t-SNE shows distinct clusters based on spatiotemporal features.

#### Feature Space Visualization (t-SNE)
| Deep Learning (2D CNN) | Deep Learning (R(2+1)D) |
|:---:|:---:|
| ![2D CNN t-SNE](results_deep_learning_2dcnn/tsne.png) | ![R(2+1)D t-SNE](results_deep_learning_r2plus1d/tsne_visualization.png) |


## 6. Deployment - Practical Considerations
- **Edge Deployment**: **2D CNN** or **Classical (if features valid)**. 2D CNN is extremely lightweight (1870 FPS theoretical).
- **Cloud Deployment**: **R(2+1)D** for robustness if compute is available.
- **Real-time**: All tested models seem capable, but 2D CNN is fastest.
