p = "scripts/build_nih_xray_manifest.py"
for i, line in enumerate(open(p), 1):
    if 'split' in line:
        print(i, repr(line.rstrip()))
