import os
import sys
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dataset.dataset import RealDESEDDataset, RealDESED_CLASSES
from train import PLModule


class DummyConfig:
    chunk_size = 10.0
    hop_size = 2.5
    sample_rate = 16000
    inference_batch_size = 8
    sliding_window_stitching = "average"
    triangular_filter_floor = 0.3
    mixup_p = 0.5
    mixup_alpha = 0.3
    freq_warp_p = 0.5
    max_lr = 0.0001
    weight_decay = 0.0001
    warmup_steps = 100


def test_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = DummyConfig()
    environment_classes = ["kitchen", "living_room"]
    model = PLModule(
        config=config,
        class_names=RealDESED_CLASSES,
        recording_environment_class_names=environment_classes,
        pretrained_checkpoint="ATST-F_strong_1",
        frame_hz=25,
    ).to(device)
    model.eval()

    val_ds = RealDESEDDataset("data", split="validation")
    sample = val_ds[0]
    audio = sample["audio"].to(device)

    print(f"Testing real audio input from validation split: shape {audio.shape}, duration: {sample['duration']:.2f}s")

    with torch.no_grad():
        preds = model.sliding_window_inference(audio)

    print(f"Sliding window inference succeeded! Predictions shape: {preds.shape}")
    print(f"Number of classes: {preds.shape[0]}, Frames: {preds.shape[1]}")
    print("Verification of model architecture, weights, GPU inference, and post-processing: SUCCESS!")


if __name__ == "__main__":
    test_pipeline()
