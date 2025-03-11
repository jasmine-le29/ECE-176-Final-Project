import kagglehub

# Download latest version
path = kagglehub.dataset_download("andy8744/udacity-self-driving-car-behavioural-cloning")

print("Path to dataset files:", path)
