import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
DOWNLOADS_DIR = DATA_DIR / "downloads"


def extract_split(split_name: str):
    zip_path = DOWNLOADS_DIR / f"{split_name}.zip"
    target_dir = DATA_DIR / split_name
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = target_dir / "audio"
    if audio_dir.exists() and (target_dir / "metadata.csv").exists():
        count = len(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac")))
        print(f"Split '{split_name}' already extracted: {count} audio files in {audio_dir}.")
        return

    if not zip_path.exists():
        print(f"Archive not found: {zip_path}")
        sys.exit(1)

    print(f"Opening archive {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total = len(members)
        print(f"Archive contains {total} files. Checking structure...")

        # Determine if members share a common root prefix like 'train/' or 'validation/'
        first_entry = members[0].filename
        prefix = ""
        if "/" in first_entry:
            candidate = first_entry.split("/")[0]
            if all(m.filename.startswith(f"{candidate}/") for m in members):
                prefix = f"{candidate}/"
                print(f"Detected common prefix in archive: '{prefix}'")

        print(f"Extracting {total} files directly into {target_dir}...")
        for idx, member in enumerate(members):
            rel_name = member.filename[len(prefix):] if prefix else member.filename
            dest_path = target_dir / rel_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

            if (idx + 1) % 500 == 0 or (idx + 1) == total:
                print(f"[{split_name}] Extracted {idx + 1}/{total} files ({(idx + 1) / total * 100:.1f}%)")

    print(f"Extraction for '{split_name}' completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_split(sys.argv[1])
    else:
        for s in ["validation", "test", "train"]:
            extract_split(s)

