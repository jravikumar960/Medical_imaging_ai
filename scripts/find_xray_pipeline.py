from pathlib import Path
root = Path("c:/Users/jravi/Downloads/medical_imaging_ai")
patterns = ["torchxrayvision", "DenseNet121", "infer_bytes", "GradCAM"]
for path in root.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if any(p in text for p in patterns):
        print(path)
