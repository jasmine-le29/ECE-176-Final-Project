import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from decision_transformer import DecisionTransformer
from dataset import get_dataloader

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecisionTransformer().to(device)
model.load_state_dict(torch.load("decision_transformer_best.pth", map_location=device))
model.eval()

# Load dataset
dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/Dataset/1/self_driving_car_dataset_make"
dataloader = get_dataloader(dataset_path, batch_size=1, seq_len=5)

# Store results
actual_steering = []
predicted_steering = []

for returns, states, actions in dataloader:
    returns, states, actions = returns.to(device), states.to(device), actions.to(device)
    predicted_actions = model(returns, states, actions).detach().cpu().numpy()

    actual_steering.append(actions.squeeze().tolist())
    predicted_steering.append(predicted_actions.squeeze().tolist())

# Flatten lists
actual_steering = [item for sublist in actual_steering for item in sublist]
predicted_steering = [item for sublist in predicted_steering for item in sublist]

# Compute errors
errors = np.abs(np.array(actual_steering) - np.array(predicted_steering))

# Metrics 
mae = np.mean(errors)
rmse = np.sqrt(np.mean(errors**2))
within_5_degrees = np.mean(errors < 0.05) * 100  # Percentage within ±5 degrees

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Percentage of Predictions Within ±5 Degrees: {within_5_degrees:.2f}%")

# Plots 
frame_indices = np.arange(len(errors)) 

# 1. Actual vs. Predicted Steering Angles
plt.figure(figsize=(10, 5))
plt.plot(actual_steering, label="Actual Steering Angles", color="blue")
plt.plot(predicted_steering, label="Predicted Steering Angles", color="red", linestyle="dashed")
plt.xlabel("Frame Index")
plt.ylabel("Steering Angle")
plt.legend()
plt.title("Actual vs. Predicted Steering Angles")
plt.show()

# 2. Histogram of Steering Predictions
plt.figure(figsize=(8, 5))
plt.hist(predicted_steering, bins=30, alpha=0.7, label="Predicted", color="red")
plt.hist(actual_steering, bins=30, alpha=0.7, label="Actual", color="blue")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Actual vs. Predicted Steering Angles")
plt.show()

# 3. Predicted vs. Actual Steering Over a Sequence
sample_index = np.random.randint(0, len(actual_steering) - 10)

plt.figure(figsize=(10, 5))
plt.plot(range(10), actual_steering[sample_index:sample_index + 10], label="Actual", marker='o', color="blue")
plt.plot(range(10), predicted_steering[sample_index:sample_index + 10], label="Predicted", marker='o', color="red", linestyle="dashed")
plt.fill_between(range(10),
    np.array(predicted_steering[sample_index:sample_index + 10]) - 0.02,  
    np.array(predicted_steering[sample_index:sample_index + 10]) + 0.02,
    color='red', alpha=0.2, label="Uncertainty")
plt.xlabel("Frame Number")
plt.ylabel("Steering Angle")
plt.legend()
plt.title("Predicted vs. Actual Steering Over a Sequence")
plt.show()

# 4. Speed vs. Steering Prediction Error (Fixing Alignment)
# Ensure speeds and errors are the same size
csv_path = f"{dataset_path}/driving_log.csv"
df = pd.read_csv(csv_path, names=["centercam", "leftcam", "rightcam", "steering_angle", "throttle", "reverse", "speed"])

min_length = min(len(errors), len(df["speed"]))

speeds = df["speed"].values[:min_length]
errors = errors[:min_length]  # Ensure both arrays are the same length

plt.figure(figsize=(8, 5))
plt.scatter(speeds, errors, alpha=0.5, color="blue")
plt.xlabel("Speed")
plt.ylabel("Prediction Error")
plt.title("Relationship Between Speed and Steering Prediction Error")
plt.show()

# 5. Boxplot of Steering Prediction Errors
plt.figure(figsize=(8, 5))
plt.boxplot(errors, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.xlabel("Error")
plt.title("Boxplot of Steering Prediction Errors")
plt.show()

# 6. Average Error Over Time (Moving Average)
window_size = 100  # Adjust window size based on your data
moving_avg_error = np.convolve(errors, np.ones(window_size) / window_size, mode="valid")

plt.figure(figsize=(10, 5))
plt.plot(moving_avg_error, color="green")
plt.xlabel("Frame Index")
plt.ylabel("Moving Average Error")
plt.title(f"Moving Average of Steering Prediction Error (Window={window_size})")
plt.show()

# 7. Kernel Density Estimate (KDE) Plot
plt.figure(figsize=(8, 5))
sns.kdeplot(predicted_steering, label="Predicted", color="red", fill=True)
sns.kdeplot(actual_steering, label="Actual", color="blue", fill=True)
plt.xlabel("Steering Angle")
plt.ylabel("Density")
plt.legend()
plt.title("KDE Plot of Actual vs. Predicted Steering Angles")
plt.show()
