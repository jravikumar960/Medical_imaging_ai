from pathlib import Path
for p in Path(".").rglob("*"):
    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg") and ".venv" not in str(p) and "xray_labeled_evaluation/images" not in str(p):
        print(p, p.stat().st_size)
