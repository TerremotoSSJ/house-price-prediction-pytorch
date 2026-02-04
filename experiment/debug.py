# Debug I used to test my model and dataset with real data
# test_real_data.py
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

print("TEST WITH REAL DATA")
print("=" * 60)

from dataset import housingDataset
from model import HousingModel

# 1. Load YOUR data
print("1. Loading real dataset...")
dataset = housingDataset()

# Verify CRITICAL statistics
print(f"\nREAL DATASET:")
print(f"  Samples: {len(dataset)}")
print(f"  Features shape: {dataset.features.shape}")
print(f"  Labels shape: {dataset.labels.shape}")

# Verify NaN/Inf
print(f"\nNaN/INF VERIFICATION:")
print(f"  Features - NaN?: {torch.isnan(dataset.features).any().item()}") #there was NaN before fix
print(f"  Features - Inf?: {torch.isinf(dataset.features).any().item()}")
print(f"  Labels - NaN?: {torch.isnan(dataset.labels).any().item()}")
print(f"  Labels - Inf?: {torch.isinf(dataset.labels).any().item()}")

# Statistics
print(f"\nSTATISTICS:")
print(f"  Features - min: {dataset.features.min().item():.6f}, max: {dataset.features.max().item():.6f}")
print(f"  Features - mean: {dataset.features.mean().item():.6f}, std: {dataset.features.std().item():.6f}")
print(f"  Labels - min: {dataset.labels.min().item():.6f}, max: {dataset.labels.max().item():.6f}")
print(f"  Labels - mean: {dataset.labels.mean().item():.6f}, std: {dataset.labels.std().item():.6f}")

# 2. Create model with YOUR data
print(f"\n2. Creating model...")
model = HousingModel(input_size=dataset.features.shape[1], base_neurons=64)

# 3. Test with a REAL sample
print(f"\n3. Testing forward pass with real data...")
sample_features, sample_label = dataset[0:5]  # 5 real samples

print(f"  Sample features shape: {sample_features.shape}")
print(f"  Sample features range: [{sample_features.min().item():.6f}, {sample_features.max().item():.6f}]")

with torch.no_grad():
    outputs = model(sample_features)
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  Outputs range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
    print(f"  Outputs - NaN?: {torch.isnan(outputs).any().item()}")

# 4. Calculate loss with real data
print(f"\n4. Testing loss with real data...")
criterion = torch.nn.MSELoss()
loss = criterion(outputs, sample_label)
print(f"  Loss: {loss.item():.6f}")
print(f"  Loss - NaN?: {torch.isnan(loss).item()}")

# 5. Complete training step
print(f"\n5. Testing a complete training step...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(sample_features)
loss = criterion(outputs, sample_label)

if torch.isnan(loss):
    print("  ERROR: Loss is NaN before backward!")
else:
    loss.backward()
    
    # Verify gradients
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  ERROR: NaN in gradients of {name}")
            has_nan_grad = True
    
    if has_nan_grad:
        print("  WARNING: NaN in gradients!")
    else:
        # Gradient clipping (IMPORTANT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print("  SUCCESS: Training step completed without NaN")

print("=" * 60)