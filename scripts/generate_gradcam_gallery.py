import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging import gradcam, prepare_image

ROOT = Path("c:/Users/jravi/Downloads/medical_imaging_ai")
MANIFEST = ROOT / "outputs/xray_labeled_evaluation/nih_xray_manifest.csv"
IMAGE_ROOT = ROOT / "outputs/xray_labeled_evaluation"
OUTPUT = ROOT / "outputs/xray_labeled_evaluation/gradcam_gallery.png"

manifest = pd.read_csv(MANIFEST)
model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(model.pathologies)
compatible = [label for label in labels if label in {"Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"}]

# Select 12 test images with the highest maximum compatible score.
test_rows = manifest[manifest["split"] == "test"].copy()
scores_for_selection = []
for _, row in test_rows.iterrows():
    path = IMAGE_ROOT / str(row["image_path"])
    with Image.open(path) as image:
        tensor, _ = prepare_image(image)
    with torch.inference_mode():
        output = model(tensor)
    sc = output[0].detach().cpu().numpy()
    compatible_scores = [sc[labels.index(label)] for label in compatible]
    scores_for_selection.append(max(compatible_scores) if compatible_scores else -1)
test_rows = test_rows.assign(_max_score=scores_for_selection).sort_values("_max_score", ascending=False).head(12)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()
for ax, (_, row) in zip(axes, test_rows.iterrows()):
    path = IMAGE_ROOT / str(row["image_path"])
    with Image.open(path) as image:
        tensor, gray = prepare_image(image)
    with torch.inference_mode():
        output = model(tensor)
    sc = output[0].detach().cpu().numpy()
    compatible_with_scores = sorted(compatible, key=lambda label: sc[labels.index(label)], reverse=True)
    top_label = compatible_with_scores[0]
    top_index = labels.index(top_label)
    top_score = float(sc[top_index])
    cam, _ = gradcam(model, tensor, top_index)
    base = np.asarray(gray.resize((224, 224)), dtype=np.float32) / 255.0
    ax.imshow(base, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.38, vmin=0, vmax=1)
    ax.axis("off")
    ref_labels = [label for label in compatible if row.get(label, 0) == 1]
    ref_text = ",".join(ref_labels) if ref_labels else "NoFinding"
    ax.set_title(f"{top_label}: {top_score:.3f}\nref: {ref_text}", fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
plt.close(fig)
print(f"Wrote {OUTPUT}")
