import cv2
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SelfDrivingDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, seq_len=5):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.dataframe) - self.seq_len)  

    def __getitem__(self, idx):
        states, actions = [], []
        for i in range(self.seq_len):
            img_path = self.dataframe.iloc[idx + i]["centercam"]
            img_filename = os.path.basename(img_path.replace("\\", "/"))  

            img_path = os.path.join(self.data_dir, "IMG", img_filename)

            # Ensure OpenCV can read the image
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = np.array(img, dtype=np.uint8)

            # Apply transformations
            if self.transform:
                img = self.transform(img)

            states.append(img)
            actions.append(self.dataframe.iloc[idx + i]["steering_angle"])

        if len(actions) == 0:
            raise ValueError("Error: actions list is empty!")

        # Convert to PyTorch tensors
        states = torch.stack(states)  
        actions = torch.tensor(actions, dtype=torch.float32)

        if actions.numel() == 0:
            returns_to_go = torch.zeros_like(actions)
        else:
            reversed_actions = np.cumsum(actions.numpy()[::-1].copy())[::-1].copy()
            returns_to_go = torch.tensor(reversed_actions, dtype=torch.float32)

        return returns_to_go, states, actions

def get_dataloader(data_path, batch_size=32, seq_len=5):
    csv_path = os.path.join(data_path, "driving_log.csv")

    column_names = ["centercam", "leftcam", "rightcam", "steering_angle", "throttle", "reverse", "speed"]

    df = pd.read_csv(csv_path, names=column_names)

    df["steering_angle"] = df["steering_angle"].astype(np.float32)

    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SelfDrivingDataset(df, data_path, transform=transform, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)