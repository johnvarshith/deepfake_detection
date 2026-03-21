import os
import sys
import json
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from preprocessing.data_loader import DeepfakeDataset, mixup_data
from models.cnn_lstm_model import CNNLSTM

def train_model(seed, model_suffix):
    """Train a model with specific seed for ensemble"""
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training Model {model_suffix} with seed {seed}")
    print(f"{'='*60}")
    
    # Dataset
    dataset = DeepfakeDataset("faces_dataset")
    
    # Different split for each model (70/15/15 but with different random splits)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Balanced sampler
    train_labels = [dataset.samples[i][1] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=8, sampler=sampler, num_workers=0, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=8, num_workers=0, pin_memory=True
    )
    
    # Model
    model = CNNLSTM().to(device)
    
    # Separate parameters
    backbone_params = []
    lstm_params = []
    fc_params = []
    
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            backbone_params.append(param)
        elif 'lstm' in name:
            lstm_params.append(param)
        else:
            fc_params.append(param)
    
    # Slightly different LRs for diversity
    lr_variations = {
        42: [2e-5, 5e-4, 1e-3],      # Model A
        99: [1.5e-5, 4e-4, 8e-4],    # Model B - slightly lower
        123: [2.5e-5, 6e-4, 1.2e-3]  # Model C - slightly higher
    }
    
    lrs = lr_variations.get(seed, [2e-5, 5e-4, 1e-3])
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lrs[0]},
        {'params': lstm_params, 'lr': lrs[1]},
        {'params': fc_params, 'lr': lrs[2]}
    ], weight_decay=1e-2)
    
    # Class-weighted loss
    real_count = sum(1 for _, l in dataset.samples if l == 0)
    fake_count = sum(1 for _, l in dataset.samples if l == 1)
    pos_weight = torch.tensor([real_count / fake_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device).unsqueeze(1).float()
                outputs = model(videos)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/deepfake_model_ensemble_{model_suffix}.pth")
            print(f"  Epoch {epoch+1}: New best val_acc: {val_acc:.3f}")
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f}")
    
    print(f"✓ Model {model_suffix} complete! Best val_acc: {best_val_acc:.3f}")
    return best_val_acc

if __name__ == "__main__":
    # Train 3 models with different seeds
    seeds = [42, 99, 123]
    suffixes = ['A', 'B', 'C']
    
    results = {}
    for seed, suffix in zip(seeds, suffixes):
        acc = train_model(seed, suffix)
        results[suffix] = acc
    
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*60)
    for model, acc in results.items():
        print(f"Model {model}: {acc:.3f} accuracy")
    print("="*60)