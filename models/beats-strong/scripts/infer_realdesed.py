"""Batch inference script for PretrainedSED (BEATs_strong_1) on RealDESED audio."""

import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_SED_DIR = PROJECT_ROOT / "PretrainedSED"
if str(PRETRAINED_SED_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINED_SED_DIR))

from data_util import audioset_classes
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper


def load_beats_strong_model(device: torch.device) -> PredictionsWrapper:
    """Load the pretrained BEATs_strong_1 model from PretrainedSED resources."""
    cwd = os.getcwd()
    try:
        os.chdir(PRETRAINED_SED_DIR)
        beats = BEATsWrapper()
        model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")
    finally:
        os.chdir(cwd)
    model.eval()
    model.to(device)
    return model


def infer_single_file(
    model: PredictionsWrapper,
    audio_path: Path,
    device: torch.device,
    thresholds=(0.1, 0.2, 0.5),
    median_window: int = 9,
    use_amp: bool = True,
):
    """Run chunked 10s inference on a single audio file and return decoded event DataFrames."""
    sample_rate = 16_000
    segment_duration = 10
    segment_samples = segment_duration * sample_rate
    frames_per_sec = 25

    waveform_np, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform_np[None, :]).to(device)
    waveform_len = waveform.shape[1]
    audio_len = waveform_len / sample_rate

    num_chunks = waveform_len // segment_samples + int(waveform_len % segment_samples != 0)
    chunks = []
    for i in range(num_chunks):
        chunk = waveform[:, i * segment_samples : min((i + 1) * segment_samples, waveform_len)]
        if chunk.shape[1] < segment_samples:
            chunk = torch.nn.functional.pad(chunk, (0, segment_samples - chunk.shape[1]))
        chunks.append(chunk)
    chunk_batch = torch.cat(chunks, dim=0)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            mel = model.mel_forward(chunk_batch)
            y_strong, _ = model(mel)

    # Concatenate chunks along time axis and trim padded frames back to actual audio length.
    y_strong = y_strong.transpose(0, 1).reshape(1, y_strong.shape[1], -1)
    true_frames = max(1, min(y_strong.shape[2], int(round(audio_len * frames_per_sec))))
    y_strong = torch.sigmoid(y_strong[:, :, :true_frames])

    encoder = ManyHotEncoder(
        audioset_classes.as_strong_train_classes,
        audio_len=true_frames / frames_per_sec,
    )
    _, _, decoded_predictions = batched_decode_preds(
        y_strong.float(),
        [audio_path.name],
        encoder,
        median_filter=median_window,
        thresholds=thresholds,
    )
    return audio_len, decoded_predictions


def main():
    parser = argparse.ArgumentParser(description="Run BEATs_strong_1 batch inference on RealDESED.")
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default=Path(r"E:\workspace\audio_team\sed\models\RealDESED\data\test\audio"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "inference_realdesed_test",
    )
    parser.add_argument("--max_files", type=int, default=0, help="0 means all files")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.5])
    parser.add_argument("--median_window", type=int, default=9)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(args.audio_dir.glob("*.wav"))
    if args.max_files > 0:
        wav_files = wav_files[: args.max_files]
    assert len(wav_files) > 0, f"No WAV files found in {args.audio_dir}"

    print(f"Device: {device} | Files: {len(wav_files)}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = load_beats_strong_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded BEATs_strong_1 ({total_params / 1e6:.2f}M parameters)")

    all_dfs = {th: [] for th in args.thresholds}
    total_audio_sec = 0.0
    t0 = time.perf_counter()

    for idx, wav_path in enumerate(wav_files, 1):
        audio_len, preds = infer_single_file(
            model,
            wav_path,
            device,
            thresholds=tuple(args.thresholds),
            median_window=args.median_window,
            use_amp=not args.no_amp,
        )
        total_audio_sec += audio_len
        for th, df in preds.items():
            if not df.empty:
                all_dfs[th].append(df)
        if idx % 100 == 0 or idx == len(wav_files):
            elapsed = time.perf_counter() - t0
            print(f"[{idx}/{len(wav_files)}] elapsed={elapsed:.1f}s ({idx / elapsed:.2f} files/s)")

    elapsed = time.perf_counter() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    )

    summary = {
        "num_files": len(wav_files),
        "total_audio_sec": round(total_audio_sec, 2),
        "elapsed_sec": round(elapsed, 2),
        "realtime_factor": round(total_audio_sec / max(elapsed, 1e-6), 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }
    print("Summary:", summary)

    for th, dfs in all_dfs.items():
        merged = (
            pd.concat(dfs, ignore_index=True)
            if dfs
            else pd.DataFrame(columns=["event_label", "onset", "offset", "filename"])
        )
        out_csv = args.output_dir / f"predictions_th{th}.csv"
        merged.to_csv(out_csv, index=False)
        top_classes = merged["event_label"].value_counts().head(15).to_dict() if not merged.empty else {}
        print(f"Saved {len(merged)} events (th={th}) -> {out_csv}")
        print(f"Top detected classes (th={th}): {top_classes}")


if __name__ == "__main__":
    main()

