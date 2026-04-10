# test_ensemble.py
import torch
import sys
sys.path.append('.')
from models.cnn_lstm_model import CNNLSTM
from models.ensemble_3models import Ensemble3Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

# Load ensemble
ensemble = Ensemble3Models(device)
print(f"Loaded {ensemble.get_model_count()} models")

# Test with random input
test_input = torch.randn(1, 20, 3, 128, 128).to(device)
print(f"\nTesting with random input (normal distribution):")
result = ensemble.predict(test_input)
print(f"Result: {result}")

# Test with ones (simulating normalized frames)
test_input = torch.ones(1, 20, 3, 128, 128).to(device)
print(f"\nTesting with ones:")
result = ensemble.predict(test_input)
print(f"Result: {result}")

# Test with zeros
test_input = torch.zeros(1, 20, 3, 128, 128).to(device)
print(f"\nTesting with zeros:")
result = ensemble.predict(test_input)
print(f"Result: {result}")