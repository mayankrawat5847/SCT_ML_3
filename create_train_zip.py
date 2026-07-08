import os
import zipfile
import numpy as np
from PIL import Image

project = r"c:\Users\Mayank Singh Rawat\Desktop\New folder\SCT_ML_3"
train_dir = os.path.join(project, "train")
os.makedirs(train_dir, exist_ok=True)

for i in range(20):
    label = "cat" if i % 2 == 0 else "dog"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    if label == "cat":
        arr[:, :, 0] = 240
        arr[8:24, 8:24] = [220, 180, 180]
        arr[10:22, 10:22] = [180, 100, 100]
    else:
        arr[:, :, 2] = 240
        arr[8:24, 8:24] = [180, 180, 220]
        arr[10:22, 10:22] = [100, 100, 180]
    path = os.path.join(train_dir, f"{label}.{i}.png")
    Image.fromarray(arr).save(path)

zip_path = os.path.join(project, "train.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(train_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, train_dir)
            z.write(full_path, arcname=arcname)

print(zip_path)
