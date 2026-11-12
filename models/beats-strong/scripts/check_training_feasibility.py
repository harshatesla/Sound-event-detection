"""Benchmark training feasibility and peak GPU VRAM for BEATs on GTX 1650 (4 GB VRAM)."""

import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_SED_DIR = PROJECT_ROOT / "PretrainedSED"
if str(PRETRAINED_SED_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINED_SED_DIR))

from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper


def build_model(mode: str, device: torch.device) -> PredictionsWrapper:
    cwd = os.getcwd()
    try:
        os.chdir(PRETRAINED_SED_DIR)
        beats = BEATsWrapper()
        if mode == "frozen_linear":
            model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1", n_classes_strong=15)
            for p in model.model.parameters():
                p.requires_grad = False
        elif mode == "frozen_bigru256":
            model = PredictionsWrapper(
                beats,
                checkpoint="BEATs_strong_1",
                n_classes_strong=15,
                seq_model_type="rnn",
                rnn_layers=2,
                rnn_dim=256,
            )
            for p in model.model.parameters():
                p.requires_grad = False
        elif mode == "last2_linear":
            model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1", n_classes_strong=15)
            for name, p in model.model.named_parameters():
                if not (".layers.10." in name or ".layers.11." in name or "beats.encoder.layer_norm" in name):
                    p.requires_grad = False
        elif mode == "full_finetune":
            model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1", n_classes_strong=15)
        else:
            raise ValueError(mode)
    finally:
        os.chdir(cwd)
    return model.to(device)


def bench(mode: str, batch_size: int, device: torch.device):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = build_model(mode, device)
        model.train()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        loss_fn = nn.BCEWithLogitsLoss()

        x = torch.randn(batch_size, 160_000, device=device)
        y = torch.zeros(batch_size, 15, 250, device=device)

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                mel = model.mel_forward(x)
                y_hat, _ = model(mel)
                loss = loss_fn(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(
            f"[PASS] mode={mode:15s} bs={batch_size:<2d} | "
            f"trainable={trainable_params/1e6:6.2f}M/{total_params/1e6:5.2f}M | peak_vram={peak_mb:7.1f} MB"
        )
    except RuntimeError as e:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[FAIL] mode={mode:15s} bs={batch_size:<2d} | peak_before_err={peak_mb:7.1f} MB | {str(e).splitlines()[0]}")
    finally:
        for var in ("model", "optimizer", "scaler", "x", "y", "mel", "y_hat", "loss"):
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
        gc.collect()


def main():
    assert torch.cuda.is_available(), "CUDA required for benchmark"
    device = torch.device("cuda")
    free_mb, total_mb = torch.cuda.mem_get_info(device)
    print(f"GPU: {torch.cuda.get_device_name(0)} | free={free_mb/(1024**2):.1f} MB / total={total_mb/(1024**2):.1f} MB")
    for mode, bs_list in [
        ("frozen_linear", [8, 16]),
        ("frozen_bigru256", [8, 16]),
        ("last2_linear", [4, 8]),
        ("full_finetune", [1, 2, 4]),
    ]:
        for bs in bs_list:
            bench(mode, bs, device)


if __name__ == "__main__":
    main()

