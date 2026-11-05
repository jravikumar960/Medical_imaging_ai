import sys
sys.path.insert(0, r"c:\Users\jravi\Downloads\medical_imaging_ai")
from pathlib import Path
import numpy as np
import torch
import torchxrayvision as xrv
from imaging import prepare_image
from PIL import Image

model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(model.pathologies)
target = "Infiltration"
candidates = []
for path in sorted(Path("outputs/xray_labeled_evaluation/images").glob("*.png")):
    with Image.open(path) as image:
        tensor, _ = prepare_image(image)
    with torch.inference_mode():
        out = model(tensor)
    scores = out[0].detach().cpu().numpy().astype(float)
    idx = labels.index(target)
    top_idx = int(np.argmax(scores))
    if labels[top_idx] == target:
        candidates.append((path.name, float(scores[idx]), float(scores[top_idx])))
candidates.sort(key=lambda x: abs(x[1] - 0.52882))
for c in candidates[:10]:
    print(c)
print("count infiltration top", len(candidates))
