import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from decision_transformer import DecisionTransformer
from dataset import get_dataloader

# Dataset Path
dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/Dataset/1/self_driving_car_dataset_make"

# Load Dataset
dataloader = get_dataloader(dataset_path, batch_size=32, seq_len=5)  # Increased batch size

# Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecisionTransformer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-5)  # Lowered learning rate
loss_fn = nn.SmoothL1Loss()  # Robust loss function

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training Settings
num_epochs = 10 
best_loss = float('inf')

# Logging Variables
losses, mae_values, rmse_values = [], [], []

# Data Augmentation Function
def augment_data(states, actions):
    if torch.rand(1) < 0.5:
        states = torch.flip(states, dims=[-1])  
        actions = -actions 
    actions += torch.randn_like(actions) * 0.01  
    return states, actions

for epoch in range(num_epochs):
    total_loss, total_mae, total_rmse, num_samples = 0, 0, 0, 0

    for returns, states, actions in dataloader:
        returns, states, actions = returns.to(device), states.to(device), actions.to(device)

        # Apply Data Augmentation
        states, actions = augment_data(states, actions)

        optimizer.zero_grad()
        predicted_actions = model(returns, states, actions)
        loss = loss_fn(predicted_actions, actions)

        # Gradient Clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

        total_loss += loss.item()
        errors = torch.abs(predicted_actions - actions)
        mae = errors.mean().item()
        rmse = torch.sqrt((errors**2).mean()).item()

        total_mae += mae * actions.size(0)
        total_rmse += rmse * actions.size(0)
        num_samples += actions.size(0)

    # Compute Average Metrics
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples

    losses.append(avg_loss)
    mae_values.append(avg_mae)
    rmse_values.append(avg_rmse)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")

    # Learning rate scheduler update
    scheduler.step(avg_loss)

    # Save Best Model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "decision_transformer_best.pth")

# Save Final Model
torch.save(model.state_dict(), "decision_transformer_final.pth")
print("Final model saved as decision_transformer_final.pth")

# Plot Training Performance
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(losses) + 1), losses, marker='o', color='blue', linestyle='-', label="Loss")
plt.plot(range(1, len(mae_values) + 1), mae_values, marker='s', color='red', linestyle='--', label="MAE")
plt.plot(range(1, len(rmse_values) + 1), rmse_values, marker='^', color='green', linestyle=':', label="RMSE")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Loss, MAE, and RMSE Over Epochs")
plt.legend()
plt.grid(True)
plt.show()





