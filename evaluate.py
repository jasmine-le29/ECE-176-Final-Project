import torch
import matplotlib.pyplot as plt
import numpy as np
from decision_transformer import DecisionTransformer
from dataset import get_dataloader

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecisionTransformer().to(device)
model.load_state_dict(torch.load("decision_transformer.pth", map_location=device))
model.eval()

dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/Dataset/1/self_driving_car_dataset_make"
dataloader = get_dataloader(dataset_path, batch_size=1, seq_len=5)

# Store results
actual_steering = []
predicted_steering = []
errors = []

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

#Plot 1: Actual vs. Predicted Steering Angles
plt.figure(figsize=(10, 5))
plt.plot(actual_steering, label="Actual Steering Angles", color="blue")
plt.plot(predicted_steering, label="Predicted Steering Angles", color="red", linestyle="dashed")
plt.xlabel("Frame Index")
plt.ylabel("Steering Angle")
plt.legend()
plt.title("Actual vs. Predicted Steering Angles")
plt.show()

#Plot 2: Steering Error Over Time
plt.figure(figsize=(10, 5))
plt.plot(errors, color="purple")
plt.xlabel("Frame Index")
plt.ylabel("Error (Absolute Difference)")
plt.title("Steering Prediction Error Over Time")
plt.show()

# Plot 3: Histogram of Steering Predictions
plt.figure(figsize=(8, 5))
plt.hist(predicted_steering, bins=30, alpha=0.7, label="Predicted", color="red")
plt.hist(actual_steering, bins=30, alpha=0.7, label="Actual", color="blue")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Actual vs. Predicted Steering Angles")
plt.show()

sample_index = 100  

plt.figure(figsize=(10, 5))
plt.plot(range(10), actual_steering[sample_index:sample_index + 10], label="Actual", marker='o', color="blue")
plt.plot(range(10), predicted_steering[sample_index:sample_index + 10], label="Predicted", marker='o', color="red", linestyle="dashed")
plt.xlabel("Frame Number")
plt.ylabel("Steering Angle")
plt.legend()
plt.title("Predicted vs. Actual Steering Over a Sequence")
plt.show()
