import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import DeepfakeDataset
from models.cnn_lstm_model import CNNLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 1. Load test dataset
# -------------------------------
dataset = DeepfakeDataset("faces_dataset")
print(f"Total samples: {len(dataset)}")

# Load test indices saved during training
indices_file = "training/splits/test_indices.json"
if not os.path.exists(indices_file):
    print("Error: test_indices.json not found. Run train_model.py first.")
    sys.exit(1)

with open(indices_file, "r") as f:
    test_indices = json.load(f)

test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
print(f"Test samples: {len(test_dataset)}")

# -------------------------------
# 2. Load ensemble models
# -------------------------------
model_paths = [
    "models/deepfake_model_ensemble_A.pth",
    "models/deepfake_model_ensemble_B.pth",
    "models/deepfake_model_ensemble_C.pth"
]

models = []
for path in model_paths:
    if os.path.exists(path):
        model = CNNLSTM().to(device)
        # Use weights_only=False because these are your own trained models
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        models.append(model)
        print(f"Loaded {path}")
    else:
        print(f"Warning: {path} not found")

if len(models) == 0:
    print("No models loaded. Exiting.")
    sys.exit(1)

print(f"Loaded {len(models)} models for ensemble.")

# -------------------------------
# 3. Collect predictions
# -------------------------------
all_labels = []
all_probs_individual = []  # list of lists: each model's probabilities
all_ensemble_probs = []

with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(device)
        labels = labels.numpy()
        all_labels.extend(labels)

        batch_probs = []
        for model in models:
            outputs = model(videos)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            batch_probs.append(probs)
        # Ensemble average
        ensemble_probs = np.mean(batch_probs, axis=0)
        all_ensemble_probs.extend(ensemble_probs)
        # Store individual model probs (for later analysis)
        for i, probs in enumerate(batch_probs):
            if len(all_probs_individual) <= i:
                all_probs_individual.append([])
            all_probs_individual[i].extend(probs)

all_labels = np.array(all_labels)
all_ensemble_probs = np.array(all_ensemble_probs)
for i in range(len(models)):
    all_probs_individual[i] = np.array(all_probs_individual[i])

# -------------------------------
# 4. Compute ensemble predictions
# -------------------------------
ensemble_preds = (all_ensemble_probs > 0.5).astype(int)

# -------------------------------
# 5. Classification report
# -------------------------------
print("\n" + "="*60)
print("ENSEMBLE EVALUATION")
print("="*60)
print(classification_report(all_labels, ensemble_preds, target_names=['Real', 'Fake']))

# -------------------------------
# 6. Confusion matrix
# -------------------------------
cm = confusion_matrix(all_labels, ensemble_preds)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Ensemble Confusion Matrix')
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
plt.savefig("training/outputs/ensemble_confusion_matrix.png")
print("\nConfusion matrix saved to training/outputs/ensemble_confusion_matrix.png")
plt.close()

# -------------------------------
# 7. ROC Curve and AUC
# -------------------------------
fpr, tpr, _ = roc_curve(all_labels, all_ensemble_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ensemble ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("training/outputs/ensemble_roc_curve.png")
print(f"ROC curve saved to training/outputs/ensemble_roc_curve.png (AUC = {roc_auc:.3f})")
plt.close()

# -------------------------------
# 8. (Optional) Compare individual models
# -------------------------------
print("\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*60)
for i in range(len(models)):
    preds = (all_probs_individual[i] > 0.5).astype(int)
    acc = np.mean(preds == all_labels)
    auc_i = roc_auc_score(all_labels, all_probs_individual[i])
    print(f"Model {chr(65+i)}: Accuracy = {acc:.4f}, AUC = {auc_i:.4f}")

# -------------------------------
# 9. Save results to JSON
# -------------------------------
results = {
    "ensemble_accuracy": float(np.mean(ensemble_preds == all_labels)),
    "ensemble_auc": float(roc_auc),
    "individual_accuracies": [float(np.mean((all_probs_individual[i] > 0.5) == all_labels)) for i in range(len(models))],
    "individual_aucs": [float(roc_auc_score(all_labels, all_probs_individual[i])) for i in range(len(models))]
}
with open("training/ensemble_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to training/ensemble_results.json")