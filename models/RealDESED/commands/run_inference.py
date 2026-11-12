import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dataset.dataset import RealDESEDDataset, RealDESED_CLASSES
from train import PLModule
from utils.evaluation import apply_median_filter, preds_to_score_df


class InferenceConfig:
    chunk_size = 10.0
    hop_size = 2.5
    sample_rate = 16000
    inference_batch_size = 8
    sliding_window_stitching = "average"
    triangular_filter_floor = 0.3
    mixup_p = 0.0
    mixup_alpha = 0.0
    freq_warp_p = 0.0
    max_lr = 0.0001
    weight_decay = 0.0001
    warmup_steps = 100


def binarize_to_events(score_df, class_names, threshold=0.5):
    """Convert frame-level score DataFrame to a list of detected event tuples."""
    events = []
    times = score_df["onset"].values
    offsets = score_df["offset"].values

    for cls in class_names:
        probs = score_df[cls].values
        active = probs >= threshold

        if not np.any(active):
            continue

        # Find contiguous True regions
        diff = np.diff(np.pad(active.astype(int), (1, 1), mode="constant"))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            onset = max(0.0, float(times[s]))
            offset = max(onset, float(offsets[e - 1]))
            conf = float(np.mean(probs[s:e]))
            events.append({
                "onset": round(onset, 3),
                "offset": round(offset, 3),
                "class": cls,
                "confidence": round(conf, 3),
            })

    events.sort(key=lambda x: (x["onset"], x["class"]))
    return events


def load_model(checkpoint_path=None, device="cuda"):
    config = InferenceConfig()
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".ckpt"):
            print(f"Loading PyTorch Lightning checkpoint from: {checkpoint_path}")
            model = PLModule.load_from_checkpoint(
                checkpoint_path,
                config=config,
                class_names=RealDESED_CLASSES,
                recording_environment_class_names=[],
                frame_hz=25,
            )
        else:
            print(f"Loading PyTorch weights from: {checkpoint_path}")
            model = PLModule(
                config=config,
                class_names=RealDESED_CLASSES,
                recording_environment_class_names=[],
                pretrained_checkpoint=checkpoint_path,
                frame_hz=25,
            )
    else:
        print("Loading default pretrained backbone: ATST-F_strong_1")
        model = PLModule(
            config=config,
            class_names=RealDESED_CLASSES,
            recording_environment_class_names=[],
            pretrained_checkpoint="ATST-F_strong_1",
            frame_hz=25,
        )

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference on the RealDESED test dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.ckpt or .pt)")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save predictions")
    parser.add_argument("--file", type=str, default=None, help="Specific audio file name (e.g. 003292.wav)")
    parser.add_argument("--limit", type=str, default="5", help="Number of files to process, or 'all'")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for event detection [0, 1]")
    parser.add_argument("--median_window", type=int, default=7, help="Median filter window in frames (7 = 280ms)")
    parser.add_argument("--save_scores", action="store_true", help="Save per-frame probability CSV files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load dataset
    print(f"Loading test split from {args.data_dir}...")
    test_ds = RealDESEDDataset(root=args.data_dir, split="test")
    print(f"Total test files available: {len(test_ds)}")

    # Filter files
    if args.file:
        indices = [i for i, row in enumerate(test_ds.metadata) if row["filename"] == args.file]
        if not indices:
            print(f"Error: File '{args.file}' not found in test metadata.")
            sys.exit(1)
    elif args.limit.lower() != "all":
        limit_num = int(args.limit)
        indices = list(range(min(limit_num, len(test_ds))))
    else:
        indices = list(range(len(test_ds)))

    print(f"Processing {len(indices)} file(s)...\n")

    # 2. Load model
    model = load_model(args.checkpoint, device=device)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_scores:
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)

    all_detected_rows = []

    # 3. Run inference
    for idx in indices:
        sample = test_ds[idx]
        filename = sample["filename"]
        audio = sample["audio"].to(device)
        duration = sample["duration"]
        events_gt = sample.get("events", [])

        with torch.no_grad():
            logits = model.sliding_window_inference(audio)
            probs = torch.sigmoid(logits)

        # Generate timestamps for score df
        frame_duration = 1.0 / model.frame_hz
        n_frames = probs.shape[1]
        ts = torch.arange(n_frames, device=device) * frame_duration

        score_df = preds_to_score_df(
            probs,
            ts,
            RealDESED_CLASSES,
            frame_hz=model.frame_hz,
        )

        # Apply median filter
        file_id = filename.replace(".wav", "")
        filtered_dict = apply_median_filter({file_id: score_df}, RealDESED_CLASSES, args.median_window)
        smooth_df = filtered_dict[file_id]

        if args.save_scores:
            smooth_df.to_csv(scores_dir / f"{file_id}.csv", index=False)

        # Binarize to events
        detected = binarize_to_events(smooth_df, RealDESED_CLASSES, threshold=args.threshold)

        for d in detected:
            all_detected_rows.append({
                "filename": filename,
                "onset": d["onset"],
                "offset": d["offset"],
                "class": d["class"],
                "confidence": d["confidence"],
            })

        # Display output
        print("=" * 70)
        print(f"File: {filename} (Duration: {duration:.2f}s)")
        print("-" * 70)
        print(f"Detected Sound Events (Threshold >= {args.threshold:.2f}, Median Window: {args.median_window}):")
        if detected:
            for ev in detected:
                print(f"  [{ev['onset']:6.2f}s - {ev['offset']:6.2f}s] {ev['class']:28s} (confidence: {ev['confidence']:.2f})")
        else:
            print("  (No sound events detected above threshold)")

        print("-" * 70)
        print("Ground Truth Annotations:")
        if events_gt:
            for gt in events_gt:
                print(f"  [{float(gt['start']):6.2f}s - {float(gt['end']):6.2f}s] {gt['label']:28s}")
        else:
            print("  (No ground truth annotations)")
        print("=" * 70 + "\n")

    # 4. Save detected events CSV
    out_csv = out_dir / "detected_events.csv"
    df_out = pd.DataFrame(all_detected_rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {len(all_detected_rows)} detected event(s) to {out_csv}")
    if args.save_scores:
        print(f"[DONE] Saved frame probability series to {scores_dir}")


if __name__ == "__main__":
    main()
