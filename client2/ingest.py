import os
import shutil
from pathlib import Path

def copy_dataset(src, dst):
    src = Path(src)
    dst = Path(dst)

    print(f"[INGEST] Copying {src} -> {dst}")

    if not src.exists():
        print(f"[ERROR] Missing path: {src}")
        return

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print("[INGEST] Copy completed.")

if __name__ == "__main__":
    CLIENT = os.getenv("CLIENT_ID")

    RAW_BASE  = f"/raw_data/{CLIENT}"
    DEST_BASE = f"/shared/data/{CLIENT}"

    # Copy Train without changing the structure
    copy_dataset(f"{RAW_BASE}/Train", f"{DEST_BASE}/Train")

    # Copy Val without changing the structure
    copy_dataset(f"{RAW_BASE}/Val", f"{DEST_BASE}/Val")

    print("[INGEST] OK.")
