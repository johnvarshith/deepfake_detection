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

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- Dataset ----------------
    dataset = DeepfakeDataset("faces_dataset")
    print(f"Total samples: {len(dataset)}")
    real_count = sum(1 for _, l in dataset.samples if l == 0)
    fake_count = sum(1 for _, l in dataset.samples if l == 1)
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    print(f"  Ratio (Real:Fake): 1:{fake_count/real_count:.2f}")

    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create train/val/test splits (70/15/15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Save indices
    os.makedirs("training/splits", exist_ok=True)
    with open("training/splits/val_indices.json", "w") as f:
        json.dump(val_dataset.indices, f)
    with open("training/splits/test_indices.json", "w") as f:
        json.dump(test_dataset.indices, f)

    # Weighted sampler for class balance
    train_labels = [dataset.samples[i][1] for i in train_dataset.indices]
    real_in_train = sum(1 for l in train_labels if l == 0)
    fake_in_train = sum(1 for l in train_labels if l == 1)
    
    print(f"\nClass distribution in training:")
    print(f"  Real: {real_in_train}")
    print(f"  Fake: {fake_in_train}")
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )

    # ---------------- Model ----------------
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

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 2e-5},
        {'params': lstm_params, 'lr': 5e-4},
        {'params': fc_params, 'lr': 1e-3}
    ], weight_decay=1e-2)

    # Class-weighted loss
    pos_weight = torch.tensor([real_count / fake_count]).to(device)
    print(f"\nPositive weight (real/fake): {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ========== FIX: NO EARLY STOPPING ==========
    epochs = 100  # Full 100 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    # ============================================

    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_val_acc = 0.0

    os.makedirs("models", exist_ok=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    print("\n" + "="*60)
    print("STARTING TRAINING WITH FULL DATASET")
    print(f"Total videos: {len(dataset)}")
    print(f"Epochs: {epochs} (NO EARLY STOPPING)")
    print("="*60 + "\n")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                # MixUp after epoch 10
                if epoch > 10 and random.random() > 0.5:
                    mixed_videos, labels_a, labels_b, lam = mixup_data(videos, labels, alpha=0.2)
                    outputs = model(mixed_videos)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    correct += (lam * (preds == labels_a).sum().item() + 
                              (1 - lam) * (preds == labels_b).sum().item())
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    correct += (preds == labels).sum().item()
                
                total += labels.size(0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                val_probs.extend(probs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        current_lr = optimizer.param_groups[0]['lr']

        # Per-class validation accuracy
        val_labels_array = np.array(val_labels_list).flatten()
        val_preds_array = (np.array(val_probs).flatten() > 0.5).astype(float)
        
        val_real_acc = (val_preds_array[val_labels_array == 0] == 0).mean() if (val_labels_array == 0).sum() > 0 else 0
        val_fake_acc = (val_preds_array[val_labels_array == 1] == 1).mean() if (val_labels_array == 1).sum() > 0 else 0

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"Real: {val_real_acc:.3f} | "
            f"Fake: {val_fake_acc:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'real_acc': val_real_acc,
                'fake_acc': val_fake_acc
            }, "models/deepfake_model_best.pth")

            print(f"  ✓ New best model! (val_acc: {val_acc:.3f})")

    # Save history
    with open("training/training_history_full.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)

    # Test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)

    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0)

    checkpoint = torch.load("models/deepfake_model_best.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(videos)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total

    test_labels = np.array(test_labels).flatten()
    test_preds = (np.array(test_probs).flatten() > 0.5).astype(float)

    real_acc = (test_preds[test_labels == 0] == 0).mean() if (test_labels == 0).sum() > 0 else 0
    fake_acc = (test_preds[test_labels == 1] == 1).mean() if (test_labels == 1).sum() > 0 else 0

    print(f"\nTest Results:")
    print(f"  Overall accuracy: {test_acc:.3f}")
    print(f"  Real accuracy: {real_acc:.3f} ({(test_labels == 0).sum()} samples)")
    print(f"  Fake accuracy: {fake_acc:.3f} ({(test_labels == 1).sum()} samples)")
    print("="*60)


if __name__ == "__main__":
    main()