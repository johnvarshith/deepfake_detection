import os
import sys
import json

# add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

from preprocessing.data_loader import DeepfakeDataset
from models.cnn_lstm_model import CNNLSTM


def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    return roc_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = DeepfakeDataset("faces_dataset")
    
    # Load validation or test indices
    # Change this to 'test_indices.json' for final evaluation
    indices_file = "training/splits/test_indices.json"
    
    if not os.path.exists(indices_file):
        print(f"Error: {indices_file} not found. Run train_model.py first.")
        return
    
    with open(indices_file, "r") as f:
        indices = json.load(f)
    
    eval_dataset = Subset(dataset, indices)
    print(f"Evaluation samples: {len(eval_dataset)}")

    loader = DataLoader(eval_dataset, batch_size=2, num_workers=0)

    # Load model
    model = CNNLSTM().to(device)
    
    model_path = "models/deepfake_model_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            
            outputs = model(videos)  # Raw logits
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs > 0.5).int().flatten()

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
            y_probs.extend(probs.numpy().flatten())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['Real', 'Fake']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'Fake'])
    plt.yticks(tick_marks, ['Real', 'Fake'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("training/outputs/confusion_matrix_fixed.png")
    plt.close()
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(y_true, y_probs, 
                              "training/outputs/roc_curve.png")
    
    print(f"\nROC curve saved to training/outputs/roc_curve.png")
    print(f"Confusion matrix saved to training/outputs/confusion_matrix_fixed.png")


if __name__ == "__main__":
    main()