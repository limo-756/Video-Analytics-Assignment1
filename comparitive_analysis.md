 Deep Learning 2D CNN model

1. Performance Comparison
The model achieved perfect performance on the test set.

Accuracy: 1.0000 (100%)
Precision: 1.0000
Recall: 1.0000
F1-Score: 1.0000
AUC: 1.00 for all classes.
Visualization Artifacts:

Confusion Matrix: results_deep_learning_2dcnn/confusion_matrix.png
ROC Curve: results_deep_learning_2dcnn/roc_curve.png
2. Computational Analysis
Training Time: ~66.4 seconds (for 10 epochs on GPU)
Inference Time: ~5.4 ms per video (Average over test set, batch size 1)
Inference Speed: ~184 FPS
Peak VRAM Usage: ~648 MB
Model Size: 11,178,051 parameters (approx. 43 MB file size)
3. Feature Analysis
t-SNE Visualization: results_deep_learning_2dcnn/tsne.png
The t-SNE plot shows clear separation between the three classes, confirming the model has learned robust distinctive features for Diving, Drumming, and Juggling.
9. Trade-off Analysis
Accuracy vs. Computational Cost: The model is highly efficient (ResNet18 backbone) while achieving perfect accuracy on this dataset. It strikes an excellent balance, being suitable for real-time applications (>30 FPS).
Data Efficiency: The model converged quickly (within 10 epochs) on the provided dataset, suggesting good data efficiency, likely due to transfer learning from ImageNet initialization.
Interpretability: While deep learning models are generally "black boxes", the t-SNE visualization demonstrates that the learned high-dimensional features are semantically meaningful and well-clustered.
