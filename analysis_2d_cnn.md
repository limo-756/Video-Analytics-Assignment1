# Deep Learning 2D CNN Model Analysis

## 1. Performance Comparison
- **Accuracy**: 1.0000 (100%)
- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1-Score**: 1.0000

The model achieves perfect classification on the test dataset across all three classes (Diving, Drumming, JugglingBalls).

### Confusion Matrix
![Confusion Matrix](results_deep_learning_2dcnn/confusion_matrix.png)

### ROC Curve
![ROC Curve](results_deep_learning_2dcnn/roc_curve.png)

## 2. Computational Analysis
- **Training Time**: ~60.68 seconds (for 10 epochs).
- **Inference Time per Video**: 0.0005 seconds (approx 1870 FPS). *(Note: This measurement may be optimistic due to asynchronous CUDA execution).*
- **Peak VRAM Usage**: 961.15 MB.
- **Model Parameters**: 11,178,051.
- **Model Architecture**: ResNet18 (pretrained) with frame averaging.

## 3. Feature Analysis
### t-SNE Visualization of Learned Representations
![t-SNE Visualization](results_deep_learning_2dcnn/tsne.png)
- The t-SNE plot shows perfect separation between the three classes. This indicates that the spatial features extracted by ResNet18, when averaged over time, form distinct clusters for each action category despite the temporal information being compressed.

## 4. Trade-off Analysis
### Accuracy vs. Computational Cost
The 2D CNN approach (Frame averaging) proves to be extremely efficient and effective for this specific dataset. It achieves perfect accuracy with lower parameter count (~11M) compared to 3D models and very fast inference times. This suggests it is a highly viable baseline for fixed-camera or distinct-background action recognition.

### Data Efficiency
The model converged very quickly (perfect validation accuracy by Epoch 1), demonstrating high data efficiency. This is largely due to the use of a pretrained ResNet18 backbone, which already possesses strong feature extraction capabilities for natural images.

### Limitations
While the performance is perfect on this test set, the "Frame Averaging" method ignores temporal ordering. It might fail on actions where temporal sequence matters (e.g., "sitting down" vs "standing up"). However, for distinct continuous actions like "Juggling" or "Drumming", spatial features are sufficient.
