from dataset import get_dataloader  # Ensure this is correct
import os
import torch

# Define dataset path
dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/udacity-dataset/1/self_driving_car_dataset_make"

dataloader = get_dataloader(dataset_path, batch_size=1, seq_len=5)

# Fetch one batch
try:
    returns, states, actions = next(iter(dataloader))

    # Print dataset info
    print("✅ Dataset Loaded Successfully!\n")
    print(f"Returns Shape: {returns.shape}")  # Expected: (1, 5)
    print(f"States Shape: {states.shape}")   # Expected: (1, 5, 3, 128, 128)
    print(f"Actions Shape: {actions.shape}")  # Expected: (1, 5)

    # Check if images exist
    img_dir = os.path.join(dataset_path, "IMG")
    if not os.path.exists(img_dir):
        print(f"❌ Error: Image directory not found: {img_dir}")
    else:
        print(f"✅ Image directory found: {img_dir}")

    # Check first image path
    sample_image_path = os.path.join(img_dir, os.listdir(img_dir)[0])
    if os.path.exists(sample_image_path):
        print(f"✅ Sample image found: {sample_image_path}")
    else:
        print(f"❌ Error: Sample image not found at: {sample_image_path}")

except Exception as e:
    print(f"❌ Dataset Loading Failed: {e}")
