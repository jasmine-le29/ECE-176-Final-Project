import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt 
from decision_transformer import DecisionTransformer
from dataset import get_dataloader

# Define dataset path
dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/Dataset/1/self_driving_car_dataset_make"

# Load dataset
dataloader = get_dataloader(dataset_path, batch_size=16, seq_len=5)

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecisionTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 30
losses = [] 
mae_values = []  # Track MAE per epoch
rmse_values = []  # Track RMSE per epoch

for epoch in range(num_epochs):
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    num_samples = 0

    for returns, states, actions in dataloader:
        returns, states, actions = returns.to(device), states.to(device), actions.to(device)

        optimizer.zero_grad()
        predicted_actions = model(returns, states, actions)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute MAE and RMSE
        errors = torch.abs(predicted_actions - actions)
        mae = errors.mean().item()
        rmse = torch.sqrt((errors**2).mean()).item()

        total_mae += mae * actions.size(0)
        total_rmse += rmse * actions.size(0)
        num_samples += actions.size(0)

    # Compute averages per epoch
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples

    losses.append(avg_loss)
    mae_values.append(avg_mae)
    rmse_values.append(avg_rmse)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")

torch.save(model.state_dict(), "decision_transformer.pth")
print("Model training complete! Model saved as decision_transformer.pth")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, marker='o', color='blue', linestyle='-', label="Loss")
plt.plot(range(1, num_epochs + 1), mae_values, marker='s', color='red', linestyle='--', label="MAE")
plt.plot(range(1, num_epochs + 1), rmse_values, marker='^', color='green', linestyle=':', label="RMSE")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Loss, MAE, and RMSE Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

