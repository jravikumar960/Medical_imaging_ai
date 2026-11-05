from pathlib import Path
root = Path("outputs")
for base in ["ct_inference", "mri_inference"]:
    print("==", base)
    for p in (root/base).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".nii", ".gz", ".png", ".json", ".csv", ".nii.gz"}:
            print(p.relative_to(root).as_posix(), p.stat().st_size)
PY