# R(2+1)D Model Analysis

## 1. Performance Comparison
- **Accuracy**: 0.9333
- **Precision**: 0.9444
- **Recall**: 0.9333
- **F1-Score**: 0.9327

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### ROC Curve
![ROC Curve](roc_curve.png)

## 2. Computational Analysis
- **Training Time**: Approx 110 seconds (for 10 epochs on GPU).
- **Inference Time per Video**: 0.0158 seconds (approx 63 FPS).
- **Model Size**: 119.50 MB.
- **Parameters**: 31,301,664.
- **Memory Requirements**: Efficient for real-time inference on standard GPUs.

## 3. Feature Analysis
### t-SNE Visualization of Learned Representations
![t-SNE Visualization](tsne_visualization.png)
- The t-SNE plot shows clear separation between the three classes (Diving, Drumming, JugglingBalls), indicating the model has learned distinct feature representations for each action.

## 4. Trade-off Analysis
### Accuracy vs. Computational Cost
The model achieves high accuracy (93.33%) with very low inference latency (15.8ms per video). This suggests an excellent trade-off, making it suitable for real-time applications where high accuracy is required without significant computational overhead during inference.

### Data Efficiency
The model achieved high performance with a relatively small fine-tuning dataset (approx 30 videos per class). This high data efficiency is attributed to transfer learning from the large-scale Kinetics-400 dataset, which allowed the model to leverage general motion features.

### Interpretability vs. Performance
While deep learning models like R(2+1)D are often considered "black boxes", the t-SNE visualization demonstrates interpretability in the feature space. The clear clusters validate that the model is making decisions based on distinct motion patterns rather than noise.
