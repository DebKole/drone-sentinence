import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 1. Dataset Configuration
# ==========================================
class MarsTripletDataset(Dataset):
    def __init__(self, image_paths, patch_size=128):
        self.image_paths = image_paths
        self.patch_size = patch_size
        
        # HEAVY Augmentations for the Positive patch to simulate what the drone sees
        # vs what the base station reference image looks like
        self.pos_transform = T.Compose([
            T.ToPILImage(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Standard transform for Anchor and Negative
        self.std_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def _get_random_crop(self, img):
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            img = cv2.resize(img, (self.patch_size, self.patch_size))
            h, w = self.patch_size, self.patch_size
            
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return img[y:y+self.patch_size, x:x+self.patch_size]

    def __getitem__(self, idx):
        # 1. Load Anchor image
        anchor_path = self.image_paths[idx]
        anchor_img = cv2.imread(anchor_path)
        if anchor_img is None:
            return self.__getitem__((idx + 1) % len(self)) # fallback
            
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        
        # 2. Extract Anchor Crop (Simulates the Reference Image)
        anchor_crop = self._get_random_crop(anchor_img)
        anchor_tensor = self.std_transform(anchor_crop)
        
        # 3. Create Positive Crop (Simulates Drone View of the SAME feature)
        # We apply heavy augmentations to teach the network to ignore lighting/blur
        positive_tensor = self.pos_transform(anchor_crop)
        
        # 4. Extract Negative Crop (Simulates a DIFFERENT part of Mars)
        neg_idx = random.randint(0, len(self.image_paths) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.image_paths) - 1)
            
        neg_img = cv2.imread(self.image_paths[neg_idx])
        if neg_img is None:
            neg_img = anchor_img 
        else:
            neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
            
        negative_crop = self._get_random_crop(neg_img)
        negative_tensor = self.std_transform(negative_crop)
        
        return anchor_tensor, positive_tensor, negative_tensor

# ==========================================
# 2. Model Definition
# ==========================================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Load MobileNetV2 features
        self.features = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        
    def forward(self, x):
        # Extract features (shape for 128x128 input is 1280x4x4)
        x = self.features(x)
        
        # Flatten the spatial dimensions so we get one massive vector per image
        x = x.view(x.size(0), -1)
        
        # L2 Normalize so we can use TripletMarginLoss properly
        x = F.normalize(x, p=2, dim=1)
        return x

# ==========================================
# 3. Training Loop
# ==========================================
def train():
    # In Kaggle, datasets are mounted in /kaggle/input
    dataset_path = "/kaggle/input"
    print(f"Scanning {dataset_path} for raw images...")
    
    # Grab all PNG and JPG files inside "images" directories
    image_paths = glob.glob(f"{dataset_path}/**/images/**/*.png", recursive=True) + \
                  glob.glob(f"{dataset_path}/**/images/**/*.jpg", recursive=True)
                  
    if not image_paths:
        print("ERROR: No images found! Check your dataset path.")
        return
        
    print(f"Found {len(image_paths)} raw Mars images. We don't need the labels!")
    
    # Take a subset if testing, or use all for full training
    # random.shuffle(image_paths)
    # image_paths = image_paths[:5000] 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = MarsTripletDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    model = SiameseNetwork().to(device)
    
    # We use TripletMarginLoss. 
    # It pulls the Anchor and Positive close together, and pushes the Negative away.
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    
    # Only train the last few layers of MobileNet to save time and prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 3 # 3 epochs is usually enough to see a massive improvement
    
    print("Starting Self-Supervised Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Wrapped dataloader in tqdm for a nice progress bar in Kaggle
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass all 3
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            # Calculate Triplet Loss
            loss = criterion(anchor_out, positive_out, negative_out)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
    # Save the custom trained weights to Kaggle's working directory so you can download it!
    output_path = "/kaggle/working/siamese_mars_weights.pth"
    torch.save(model.features.state_dict(), output_path)
    print(f"Training Complete! Saved weights to {output_path}")

if __name__ == "__main__":
    train()
