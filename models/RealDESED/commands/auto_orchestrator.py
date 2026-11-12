import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
DATA_DIR = ROOT_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"

SPLIT_INFO = {
    "validation": {
        "zip": DOWNLOADS_DIR / "validation.zip",
        "expected_bytes": 1586397692,
        "marker": DATA_DIR / "validation" / "metadata.csv",
    },
    "test": {
        "zip": DOWNLOADS_DIR / "test.zip",
        "expected_bytes": 1508251418,
        "marker": DATA_DIR / "test" / "metadata.csv",
    },
    "train": {
        "zip": DOWNLOADS_DIR / "train.zip",
        "expected_bytes": 5647895421,
        "marker": DATA_DIR / "train" / "metadata.csv",
    },
}


def is_zip_complete(zip_path: Path, expected_size: int) -> bool:
    if not zip_path.exists():
        return False
    size = zip_path.stat().st_size
    if size < expected_size:
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_file = zf.testzip()
            return bad_file is None
    except Exception:
        return False


def extract_split(split_name: str):
    info = SPLIT_INFO[split_name]
    zip_path = info["zip"]
    target_dir = DATA_DIR / split_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if info["marker"].exists() and (target_dir / "audio").exists():
        print(f"Split '{split_name}' already extracted.")
        return

    print(f"\n[EXTRACTING] {zip_path} into {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total = len(members)

        first_entry = members[0].filename
        prefix = ""
        if "/" in first_entry:
            candidate = first_entry.split("/")[0]
            if all(m.filename.startswith(f"{candidate}/") for m in members):
                prefix = f"{candidate}/"

        for idx, member in enumerate(members):
            rel_name = member.filename[len(prefix):] if prefix else member.filename
            dest_path = target_dir / rel_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

            if (idx + 1) % 500 == 0 or (idx + 1) == total:
                print(f"[{split_name}] Extracted {idx + 1}/{total} files ({(idx + 1) / total * 100:.1f}%)")

    print(f"[COMPLETED] Extraction of '{split_name}'.")


def main():
    extracted = set()

    # Loop until all three splits are extracted
    while len(extracted) < 3:
        for split, info in SPLIT_INFO.items():
            if split in extracted:
                continue

            if info["marker"].exists() and (DATA_DIR / split / "audio").exists():
                print(f"Split '{split}' is already present on disk.")
                extracted.add(split)
                continue

            if is_zip_complete(info["zip"], info["expected_bytes"]):
                print(f"Archive '{split}.zip' download complete! Extracting...")
                extract_split(split)
                extracted.add(split)

        if len(extracted) < 3:
            time.sleep(10)

    print("\nAll dataset splits have been downloaded and extracted successfully!")

    # Step 1: Compute annotator quality scores
    print("\nRunning compute_annotator_score.py...")
    subprocess.run(
        [str(VENV_PYTHON), str(ROOT_DIR / "compute_annotator_score.py"), "--root", str(DATA_DIR)],
        check=True,
        cwd=ROOT_DIR,
    )

    print("\nDataset ready and annotator scores computed!")


if __name__ == "__main__":
    main()

