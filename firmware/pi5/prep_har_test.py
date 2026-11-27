import os
import urllib.request
import zipfile
import numpy as np

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH = "data/raw/uci_har.zip"
OUT_PATH = "data/raw/har_test.npz"

if not os.path.exists(ZIP_PATH):
    print("Downloading UCI HAR dataset (~60 MB)...")
    urllib.request.urlretrieve(URL, ZIP_PATH)

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    channels = []
    for sig in SIGNALS:
        path = f"UCI HAR Dataset/test/Inertial Signals/{sig}_test.txt"
        with zf.open(path) as f:
            channels.append(np.loadtxt(f, dtype=np.float32))
    X_test = np.stack(channels, axis=-1)  # (2947, 128, 9)

    with zf.open("UCI HAR Dataset/test/y_test.txt") as f:
        y_test = np.loadtxt(f, dtype=np.int32) - 1  # 0..5

np.savez_compressed(OUT_PATH, X_test=X_test, y_test=y_test)
os.remove(ZIP_PATH)
print(f"Saved {OUT_PATH}: X_test={X_test.shape}, y_test={y_test.shape}")
