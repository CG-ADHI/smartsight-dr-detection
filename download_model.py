import gdown
import os

# 🔴 REPLACE this with YOUR file ID
FILE_ID = "10UJ5WxPFEvpeZkHH8qChdr3Ra7srQBNu"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("Model already exists")
