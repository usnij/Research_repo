import os, shutil

SRC_DIR = r"C:\Users\User\Documents\blender_lender\saebit\High"     #cameragroup dir
DST_DIR = r"C:\Users\User\Documents\blender_lender\saebit"              

os.makedirs(DST_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    base = os.path.splitext(fname)[0]  # f0001-Low_Cam01
    if "-" not in base:
        print("Skip (no '-'): ", fname)
        continue

    cam_name = base.split("-")[-1]     # Low_Cam01
    cam_dir = os.path.join(DST_DIR, cam_name)
    os.makedirs(cam_dir, exist_ok=True)

    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(cam_dir, fname)

    shutil.copy2(src_path, dst_path)

print("Done. Organized by camera.")