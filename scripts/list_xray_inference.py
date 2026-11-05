from pathlib import Path
for p in Path("outputs/xray_inference").rglob("*"):
    if p.is_file():
        print(p, p.stat().st_size)
