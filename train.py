import torch
import torch.optim as optim
import torch.nn as nn
from models.decision_transformer import DecisionTransformer
from dataset import get_dataloader

dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/udacity-dataset/1/self_driving_car_dataset_make"

# Load dataset using DataLoader
dataloader = get_dataloader(dataset_path, batch_size=16, seq_len=5)

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecisionTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for returns, states, actions in dataloader:
        returns, states, actions = returns.to(device), states.to(device), actions.to(device)

        optimizer.zero_grad()
        predicted_actions = model(returns, states, actions)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "decision_transformer.pth")
print("✅ Model training complete! Model saved as decision_transformer.pth")
