import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DOWNLOADS_DIR = DATA_DIR / "downloads"

SPLITS = [
    {
        "name": "validation",
        "url": "https://zenodo.org/records/20056072/files/validation.zip?download=1",
        "zip_path": DOWNLOADS_DIR / "validation.zip",
        "expected_files": ["metadata.csv", "annotations.csv"],
    },
    {
        "name": "test",
        "url": "https://zenodo.org/records/20056072/files/test.zip?download=1",
        "zip_path": DOWNLOADS_DIR / "test.zip",
        "expected_files": ["metadata.csv", "annotations.csv"],
    },
    {
        "name": "train",
        "url": "https://zenodo.org/records/20056072/files/train.zip?download=1",
        "zip_path": DOWNLOADS_DIR / "train.zip",
        "expected_files": ["metadata.csv", "annotations_raw.csv"],
    },
]


def download_with_progress(url: str, output_path: Path):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"File already exists: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB), skipping download.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download_path = output_path.with_suffix(".part")

    print(f"Downloading from {url} to {output_path}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RealDESED-Reproduction-Script/1.0"}
    )
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 4  # 4 MB chunks
        last_percent = -1

        with open(temp_download_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = int((downloaded / total_size) * 100)
                    if percent != last_percent and percent % 5 == 0:
                        print(f"[{output_path.name}] Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent}%)")
                        last_percent = percent

    # Atomic rename once download is verified complete
    temp_download_path.rename(output_path)
    print(f"Finished downloading {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB).")


def extract_split(zip_path: Path, split_name: str, expected_files: list):
    target_dir = DATA_DIR / split_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    audio_dir = target_dir / "audio"
    if audio_dir.exists() and all((target_dir / f).exists() for f in expected_files):
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac"))
        if len(audio_files) > 0:
            print(f"Split '{split_name}' already extracted at {target_dir} ({len(audio_files)} audio files). Skipping extraction.")
            return

    print(f"Extracting {zip_path} for split '{split_name}' into {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_members = len(members)

        # Check if archive entries have prefix e.g. "train/" or "train_split/"
        # Find prefix if common to all files
        first_part = members[0].filename.split("/")[0] if "/" in members[0].filename else ""
        has_common_prefix = bool(first_part) and all(
            m.filename.startswith(f"{first_part}/") or m.filename == first_part
            for m in members if not m.is_dir()
        )

        for idx, member in enumerate(members):
            filename = member.filename
            if member.is_dir():
                continue

            if has_common_prefix:
                rel_path = filename[len(first_part) + 1 :]
            else:
                rel_path = filename

            dest_path = target_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

            if (idx + 1) % 500 == 0 or idx + 1 == total_members:
                print(f"[{split_name}] Extracted {idx + 1}/{total_members} files...")

    print(f"Extraction complete for '{split_name}'.")


def verify_dataset_layout():
    print("\nVerifying dataset layout...")
    all_ok = True
    for item in SPLITS:
        split_name = item["name"]
        split_dir = DATA_DIR / split_name
        audio_dir = split_dir / "audio"
        if not split_dir.exists():
            print(f"ERROR: Missing split directory: {split_dir}")
            all_ok = False
            continue
        if not audio_dir.exists():
            print(f"ERROR: Missing audio directory: {audio_dir}")
            all_ok = False
        else:
            audio_count = len(list(audio_dir.iterdir()))
            print(f"Split '{split_name}': {audio_count} audio files in {audio_dir}")

        for expected in item["expected_files"]:
            p = split_dir / expected
            if not p.exists():
                print(f"ERROR: Missing expected file: {p}")
                all_ok = False
            else:
                print(f"Split '{split_name}': Found {expected} ({p.stat().st_size} bytes)")

    if all_ok:
        print("\nAll dataset splits verified successfully!")
    else:
        print("\nDataset verification failed!")
    return all_ok


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    for item in SPLITS:
        download_with_progress(item["url"], item["zip_path"])
        extract_split(item["zip_path"], item["name"], item["expected_files"])

    if not verify_dataset_layout():
        sys.exit(1)


if __name__ == "__main__":
    main()

