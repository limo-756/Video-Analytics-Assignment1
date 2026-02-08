
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
import os
import time

# Set device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# Define transforms
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
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

        if not os.path.exists(video_path):
             if video_path.startswith("./"):
                 video_path = video_path[2:]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
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
        self.pooling = pooling
        self.dropout = nn.Dropout(dropped_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, T, C, H, W = x.shape
        x = x.view(batch_size * T, C, H, W)
        features = self.backbone(x)
        features = features.view(batch_size, T, 512)
        if self.pooling == 'avg':
            combined = torch.mean(features, dim=1)
        else:
            combined, _ = torch.max(features, dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)

def main():
    test_df = pd.read_csv("./dataset/splits/test.csv", index_col='index')
    test_dataset = VideoDataset(test_df, 16, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2) # Batch size 1 for per-video timing

    model = VideoClassifier(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Warmup
    print("Warming up...")
    dummy = torch.randn(1, 16, 3, 224, 224).to(DEVICE)
    for _ in range(5):
        _ = model(dummy)
    
    print("Measuring inference time...")
    times = []
    with torch.no_grad():
        for videos, _ in test_loader:
            videos = videos.to(DEVICE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            _ = model(videos)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            
    avg_inference_time = np.mean(times)
    print(f"Average Inference Time per Video (Corrected): {avg_inference_time:.4f} seconds")
    print(f"FPS: {1.0/avg_inference_time:.2f}")

if __name__ == "__main__":
    main()
