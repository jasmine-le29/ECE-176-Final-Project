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

# --- New Metrics ---
mae = np.mean(errors)
rmse = np.sqrt(np.mean(errors**2))
within_5_degrees = np.mean(errors < 0.05) * 100

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Percentage of Predictions Within ±5 Degrees: {within_5_degrees:.2f}%")

# --- New Graphs ---
# Cumulative Steering Error Over Time
cumulative_error = np.cumsum(errors)
plt.figure(figsize=(10, 5))
plt.plot(cumulative_error, color="green")
plt.xlabel("Frame Index")
plt.ylabel("Cumulative Error")
plt.title("Cumulative Steering Error Over Time")
plt.show()

# Error Distribution Histogram
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, alpha=0.7, color="purple")
plt.xlabel("Steering Error")
plt.ylabel("Frequency")
plt.title("Distribution of Steering Errors")
plt.show()

