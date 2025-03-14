import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "/Users/jasminele/Documents/decision_transformer_self_driving/Dataset/1/self_driving_car_dataset_make"
csv_path = os.path.join(dataset_path, "driving_log.csv")

column_names = ["centercam", "leftcam", "rightcam", "steering_angle", "throttle", "reverse", "speed"]

# Load CSV
df = pd.read_csv(csv_path, names=column_names)

sample_indices = [10, 100, 500]  
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, idx in enumerate(sample_indices):
    img_filename = os.path.basename(df.iloc[idx]["centercam"].replace("\\", "/"))
    img_path = os.path.join(dataset_path, "IMG", img_filename)

    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i+1} (Steering: {df.iloc[idx]['steering_angle']:.2f})")
        axes[i].axis("off")
    else:
        print(f"Error loading image: {img_path}")

plt.suptitle("Sample Images from Udacity Self-Driving Dataset")
plt.show()
