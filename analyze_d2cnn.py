
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# Set device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# Define transforms (Same as notebook/train_improved)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # Note: Notebook used Resize((224,224)) for test
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoDataset(Dataset):
    def __init__(self, df, num_frames=16, transform=None):
        self.df = df
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['clip_path']
        label = self.df.iloc[idx]['encoded_label']

        # Fix path if needed
        if not os.path.exists(video_path):
             if video_path.startswith("./"):
                 video_path = video_path[2:]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
             # Warning omitted to reduce log noise during analysis
             return torch.zeros(self.num_frames, 3, 224, 224), torch.tensor(label, dtype=torch.long)

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
        cap.release()
        
        while len(frames) < self.num_frames:
             if len(frames) > 0:
                frames.append(frames[-1])
             else:
                return torch.zeros(self.num_frames, 3, 224, 224), torch.tensor(label, dtype=torch.long)

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3, pooling='avg', dropped_rate=0.5):
        super(VideoClassifier, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze/Unfreeze logic as per train_improved
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[6:].parameters():
            param.requires_grad = True

        self.pooling = pooling
        self.dropout = nn.Dropout(dropped_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        # x shape: (Batch, Frames, C, H, W)
        batch_size, T, C, H, W = x.shape
        x = x.view(batch_size * T, C, H, W)

        features = self.backbone(x)
        features = features.view(batch_size, T, 512)

        if self.pooling == 'avg':
            combined = torch.mean(features, dim=1)
        else:
            combined, _ = torch.max(features, dim=1)
        
        if return_features:
            return combined

        combined = self.dropout(combined)
        return self.fc(combined)

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    best_acc = 0.0
    
    # Timing
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for videos, labels in train_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        val_acc, _, _, _ = evaluate_model(model, val_loader)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    total_time = time.time() - start_time
    return total_time

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # 1. Setup Data
    train_df = pd.read_csv("./dataset/splits/train.csv", index_col='index')
    val_df = pd.read_csv("./dataset/splits/validation.csv", index_col='index')
    test_df = pd.read_csv("./dataset/splits/test.csv", index_col='index')
    
    # Verify classes
    # Assuming standard mapping: 'Diving': 0, 'Drumming': 1, 'JugglingBalls': 2
    # But usually encoded_label is present.
    classes = ['Diving', 'Drumming', 'JugglingBalls'] # Based on prompt description
    
    dataset = VideoDataset(train_df, 16, transform=train_transforms)
    val_dataset = VideoDataset(val_df, 16, transform=test_transforms)
    test_dataset = VideoDataset(test_df, 16, transform=test_transforms)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 2. Model & Computational Analysis (Memory & Params)
    print("--- Computational Analysis ---")
    model = VideoClassifier(num_classes=3).to(DEVICE)
    params = count_parameters(model)
    print(f"Model Parameters: {params}")
    
    # Measure memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    # 3. Training & Timing
    print("\n--- Training ---")
    training_time = train_model(model, train_loader, val_loader, epochs=10)
    print(f"Total Training Time: {training_time:.2f} seconds")
    
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM during training: {max_mem:.2f} MB")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 4. Inference Time Analysis
    print("\n--- Inference Speed Test ---")
    model.eval()
    times = []
    with torch.no_grad():
        for videos, _ in test_loader:
            videos = videos.to(DEVICE)
            start = time.time()
            _ = model(videos)
            end = time.time()
            # Batch size is 4, so time per video is (end-start)/4
            times.append((end-start) / videos.size(0))
    
    avg_inference_time = np.mean(times)
    print(f"Average Inference Time per Video: {avg_inference_time:.4f} seconds")
    print(f"FPS: {1.0/avg_inference_time:.2f}")

    # 5. Performance Analysis
    print("\n--- Performance Analysis ---")
    acc, y_true, y_pred, y_prob = evaluate_model(model, test_loader)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Save CM Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig('results_deep_learning_2dcnn/confusion_matrix.png')
    plt.close()
    
    # ROC / AUC
    # Binarize labels for ROC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]
    
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (area = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('results_deep_learning_2dcnn/roc_curve.png')
    plt.close()

    # 6. Feature Analysis (t-SNE)
    print("\n--- Feature Analysis (t-SNE) ---")
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(DEVICE)
            feats = model(videos, return_features=True)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    tsne = TSNE(n_components=2, perplexity=min(5, features.shape[0]-1), random_state=42)
    features_embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(8,8))
    for i, c in enumerate(classes):
        idx = labels == i
        plt.scatter(features_embedded[idx, 0], features_embedded[idx, 1], label=c, alpha=0.6)
    
    plt.legend()
    plt.title('t-SNE of Learned Features')
    plt.savefig('results_deep_learning_2dcnn/tsne.png')
    plt.close()
    
    print("All results saved to results_deep_learning_2dcnn/")

if __name__ == "__main__":
    main()
