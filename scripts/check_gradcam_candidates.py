import sys
sys.path.insert(0, r"c:\Users\jravi\Downloads\medical_imaging_ai")
from pathlib import Path
import torch
import torchxrayvision as xrv
from imaging import prepare_image
from PIL import Image

model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(model.pathologies)
for path in [Path("outputs/xray_labeled_evaluation/images/1021_0.png"), Path("outputs/first_inference/chest_xray_kaggle_normal.jpeg")]:
    with Image.open(path) as image:
        tensor, _ = prepare_image(image)
    with torch.inference_mode():
        out = model(tensor)
    scores = out[0].detach().cpu().numpy()
    order = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)[:3]
    print(path.name, [(l, round(float(s), 5)) for l, s in order])
