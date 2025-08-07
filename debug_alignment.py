#!/usr/bin/env python3
"""Test to verify data alignment issue in averaging script"""
import pandas as pd
import numpy as np
import sys
import os

print("Testing data alignment...")

# Simulate the issue
# Original data: 1000 samples
original_size = 1000
y_original = np.random.randn(original_size)

# Apply balancing (similar to the script)
pos_idx = np.where(y_original > 0)[0]
neg_idx = np.where(y_original <= 0)[0]
min_count = min(len(pos_idx), len(neg_idx))

print(f"Original size: {original_size}")
print(f"Positive samples: {len(pos_idx)}")
print(f"Negative samples: {len(neg_idx)}")
print(f"After balancing: {min_count * 2}")

np.random.seed(42)
pos_sample = np.random.choice(pos_idx, min_count, replace=False)
neg_sample = np.random.choice(neg_idx, min_count, replace=False)
balanced_idx = np.concatenate([pos_sample, neg_sample])
np.random.shuffle(balanced_idx)

y_balanced = y_original[balanced_idx]

print(f"Balanced y_train size: {len(y_balanced)}")

# Now simulate getting seq_indices from original data
seq_idx = np.arange(original_size)  # This represents train_seq_idx from original data

print(f"Sequence indices size: {len(seq_idx)}")

# The issue: we try to align seq_idx (from original data) with y_balanced (from balanced data)
if len(seq_idx) > len(y_balanced):
    seq_idx_truncated = seq_idx[:len(y_balanced)]
    print(f"Truncated seq_idx size: {len(seq_idx_truncated)}")
    
    # This creates a mismatch!
    print(f"seq_idx[0:5]: {seq_idx_truncated[:5]}")
    print(f"balanced_idx[0:5]: {balanced_idx[:5]}")
    print(f"These don't match! This causes the data alignment issue.")

print("\nThe problem: we're using indices [0,1,2,3,4...] to get ActualAvg from original data,")
print("but y_train comes from balanced_idx which might be [45,123,7,891,234...]")
print("So ActualAvg and Actual (y_train) are from completely different rows!")
