import os
import glob
import pathlib

root = "data"
if not os.path.exists(root):
    print("data folder not found")
    raise SystemExit(1)

folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and os.path.exists(os.path.join(root, f, "input.json"))]
print("detected_folders=", folders)
for f in folders:
    p = pathlib.Path(root) / f
    print("---", f)
    try:
        print(" input.json content:")
        print(open(p / "input.json", encoding="utf-8").read())
    except Exception as e:
        print("  read error", e)
    excels = glob.glob(str(p / "*.xlsx"))
    print(" excel files:", excels)
