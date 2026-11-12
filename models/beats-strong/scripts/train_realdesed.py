"""Fine-tune and evaluate BEATs (PretrainedSED) on RealDESED Sound Event Detection."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import sed_scores_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_SED_DIR = PROJECT_ROOT / "PretrainedSED"
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(PRETRAINED_SED_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers.augment import (
    RandomResizeCrop,
    filter_augmentation,
    frame_shift,
    mixstyle,
    mixup,
    time_mask,
)
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from helpers.utils import worker_init_fn
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper
from realdesed_dataset import (
    REALDESED_CLASSES,
    RealDESEDEvalDataset,
    RealDESEDTrainDataset,
    RealDESEDWavMixDataset,
    eval_collate_identity,
)


def _checkpoint_name(pretrained: str) -> str:
    mapping = {
        "scratch": None,
        "ssl": "BEATs_ssl",
        "weak": "BEATs_weak",
        "strong": "BEATs_strong_1",
    }
    if pretrained not in mapping:
        raise ValueError(f"Unknown pretrained checkpoint option: {pretrained}")
    return mapping[pretrained]


class RealDESEDPLModule(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning BEATs on RealDESED."""

    def __init__(self, config: argparse.Namespace = None, **kwargs):
        super().__init__()
        if config is None:
            config = argparse.Namespace(**kwargs)
        self.save_hyperparameters(vars(config))
        self.config = config
        self.classes = REALDESED_CLASSES
        self.train_encoder = ManyHotEncoder(
            self.classes, audio_len=10.0, frame_hop=160, net_pooling=4, fs=16_000
        )

        cwd = os.getcwd()
        try:
            os.chdir(PRETRAINED_SED_DIR)
            beats = BEATsWrapper()
            seq_model_type = "rnn" if config.train_mode == "frozen_bigru" else None
            self.model = PredictionsWrapper(
                beats,
                checkpoint=_checkpoint_name(config.pretrained),
                n_classes_strong=len(self.classes),
                seq_model_type=seq_model_type,
                rnn_layers=2,
                rnn_dim=config.rnn_dim,
                rnn_dropout=config.rnn_dropout,
            )
        finally:
            os.chdir(cwd)

        self._configure_trainable_params(config.train_mode)
        self.strong_loss = nn.BCEWithLogitsLoss()
        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self._eval_scores: Dict[str, object] = {}
        self._eval_gt: Dict[str, List] = {}
        self._eval_durations: Dict[str, float] = {}
        self._test_decoded: Dict[float, List[pd.DataFrame]] = {0.3: [], 0.5: []}
        self.output_dir: Path = PROJECT_ROOT / "outputs" / f"train_{config.run_name}"

    def _configure_trainable_params(self, train_mode: str):
        # Freeze weak_head in all modes because RealDESED trains on strong frame-level labels only.
        for p in self.model.weak_head.parameters():
            p.requires_grad = False

        if train_mode == "frozen_linear":
            for p in self.model.model.parameters():
                p.requires_grad = False
        elif train_mode == "frozen_bigru":
            for p in self.model.model.parameters():
                p.requires_grad = False
        elif train_mode == "last2_linear":
            for name, p in self.model.model.named_parameters():
                keep = (
                    ".layers.10." in name
                    or ".layers.11." in name
                    or "beats.encoder.layer_norm" in name
                )
                p.requires_grad = keep
        elif train_mode == "full":
            for p in self.model.model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unsupported train_mode: {train_mode}")

    def on_train_epoch_start(self):
        if self.config.train_mode in ("frozen_linear", "frozen_bigru"):
            self.model.model.eval()

    def configure_optimizers(self):
        head_params = [p for p in self.model.strong_head.parameters() if p.requires_grad]
        seq_params = [p for p in self.model.seq_model.parameters() if p.requires_grad]

        param_groups = [{"params": head_params, "lr": self.config.max_lr}]
        if seq_params:
            param_groups.append({"params": seq_params, "lr": self.config.max_lr})

        transformer_lr = self.config.transformer_lr or self.config.max_lr
        if self.config.train_mode in ("last2_linear", "full"):
            pt_layers = self.model.separate_params()
            for depth_idx, layer_params in enumerate(pt_layers, start=1):
                trainable = [p for p in layer_params if p.requires_grad]
                if trainable:
                    lr_scaled = transformer_lr * (self.config.lr_decay ** depth_idx)
                    param_groups.append({"params": trainable, "lr": lr_scaled})

        split_groups = []
        for group in param_groups:
            params_2d = [p for p in group["params"] if p.ndimension() >= 2]
            params_1d = [p for p in group["params"] if p.ndimension() <= 1]
            if params_2d:
                split_groups.append(
                    {"params": params_2d, "lr": group["lr"], "weight_decay": self.config.weight_decay}
                )
            if params_1d:
                split_groups.append({"params": params_1d, "lr": group["lr"], "weight_decay": 0.0})

        if self.config.weight_decay > 0 or not self.config.no_adamw:
            optimizer = torch.optim.AdamW(split_groups, lr=self.config.max_lr)
        else:
            optimizer = torch.optim.Adam(split_groups, lr=self.config.max_lr)

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = min(self.config.warmup_steps, max(1, total_steps // 10))
        if self.config.schedule_mode in ("cos", "cosine"):
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif self.config.schedule_mode == "linear":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                power=1.0,
                lr_end=self.config.lr_end,
            )
        else:
            raise ValueError(f"Unsupported schedule_mode: {self.config.schedule_mode}")

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        audios = batch["audio"]
        labels = batch["strong"]

        if self.config.train_mode in ("frozen_linear", "frozen_bigru"):
            self.model.model.eval()

        mel = self.model.mel_forward(audios)

        if self.config.frame_shift_range > 0:
            (mel, labels) = frame_shift(
                mel,
                labels,
                net_pooling=self.train_encoder.net_pooling,
                shift_range=self.config.frame_shift_range,
            )

        if self.config.mixup_p > random.random():
            (mel, labels) = mixup(mel, targets=labels)

        if self.config.mixstyle_p > random.random():
            mel = mixstyle(mel)

        if self.config.max_time_mask_size > 0:
            (mel, labels) = time_mask(
                mel,
                labels,
                net_pooling=self.train_encoder.net_pooling,
                max_mask_ratio=self.config.max_time_mask_size,
            )

        if self.config.filter_augment_p > random.random():
            mel, _ = filter_augmentation(mel)

        if self.config.freq_warp_p > random.random():
            mel = self.freq_warp(mel.squeeze(1)).unsqueeze(1)

        y_hat_strong, _ = self.model(mel)
        loss = self.strong_loss(y_hat_strong, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=audios.size(0))
        return loss

    def _infer_full_clip(self, waveform: torch.Tensor, audio_len: float) -> torch.Tensor:
        segment_samples = 10 * 16_000
        frames_per_sec = 25
        waveform = waveform.unsqueeze(0).to(self.device)
        wlen = waveform.shape[1]
        num_chunks = wlen // segment_samples + int(wlen % segment_samples != 0)
        chunks = []
        for i in range(num_chunks):
            c = waveform[:, i * segment_samples : min((i + 1) * segment_samples, wlen)]
            if c.shape[1] < segment_samples:
                c = torch.nn.functional.pad(c, (0, segment_samples - c.shape[1]))
            chunks.append(c)
        chunk_batch = torch.cat(chunks, dim=0)

        mel = self.model.mel_forward(chunk_batch)
        y_strong, _ = self.model(mel)
        y_strong = y_strong.transpose(0, 1).reshape(1, y_strong.shape[1], -1)
        true_frames = max(1, min(y_strong.shape[2], int(round(audio_len * frames_per_sec))))
        return torch.sigmoid(y_strong[:, :, :true_frames]).float()

    def _eval_step(self, batch: List[dict], is_test: bool = False):
        for sample in batch:
            audio_id = Path(sample["filename"]).stem
            probs = self._infer_full_clip(sample["audio"], sample["audio_len"])
            true_dur = probs.shape[2] / 25.0
            clip_encoder = ManyHotEncoder(
                self.classes, audio_len=true_dur, frame_hop=160, net_pooling=4, fs=16_000
            )
            thresholds = (0.3, 0.5) if is_test else ()
            _, scores_postprocessed, decoded_dfs = batched_decode_preds(
                probs,
                [sample["filename"]],
                clip_encoder,
                thresholds=thresholds,
                median_filter=self.config.median_window,
            )
            self._eval_scores[audio_id] = scores_postprocessed[audio_id]
            self._eval_gt[audio_id] = sample["events"]
            self._eval_durations[audio_id] = true_dur
            if is_test:
                for th, df in decoded_dfs.items():
                    if not df.empty:
                        self._test_decoded[th].append(df)

    def _eval_epoch_end(self, prefix: str):
        # Drop classes with zero ground truth events only if a small sanity subset lacks some class.
        present_classes = {e[2] for events in self._eval_gt.values() for e in events}
        if len(present_classes) < len(self.classes):
            unused = list(set(self.classes) - present_classes)
            for df in self._eval_scores.values():
                df.drop(columns=unused, axis=1, inplace=True)

        psds1_res = sed_scores_eval.intersection_based.psds(
            self._eval_scores,
            self._eval_gt,
            self._eval_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1,
        )
        pauroc_res = sed_scores_eval.segment_based.auroc(
            self._eval_scores,
            self._eval_gt,
            self._eval_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1,
        )

        per_class_psds1 = {k: float(v) for k, v in psds1_res[1].items()}
        macro_psds1 = float(np.mean(list(per_class_psds1.values())))
        metrics = {
            f"{prefix}/psds1": float(psds1_res[0]),
            f"{prefix}/psds1_macro": macro_psds1,
            f"{prefix}/pauroc": float(pauroc_res[0]["mean"]),
        }

        if prefix == "test":
            psds2_res = sed_scores_eval.intersection_based.psds(
                self._eval_scores,
                self._eval_gt,
                self._eval_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                num_jobs=1,
            )
            per_class_psds2 = {k: float(v) for k, v in psds2_res[1].items()}
            metrics["test/psds2"] = float(psds2_res[0])
            metrics["test/psds2_macro"] = float(np.mean(list(per_class_psds2.values())))
            self.test_report = {
                **metrics,
                "per_class_psds1": per_class_psds1,
                "per_class_psds2": per_class_psds2,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for th, dfs in self._test_decoded.items():
                merged = (
                    pd.concat(dfs, ignore_index=True)
                    if dfs
                    else pd.DataFrame(columns=["event_label", "onset", "offset", "filename"])
                )
                merged.to_csv(self.output_dir / f"test_predictions_th{th}.csv", index=False)
            self._test_decoded = {0.3: [], 0.5: []}

        self.log_dict(metrics, prog_bar=True, sync_dist=False)

        self._eval_scores = {}
        self._eval_gt = {}
        self._eval_durations = {}

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, is_test=False)

    def on_validation_epoch_end(self):
        self._eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, is_test=True)

    def on_test_epoch_end(self):
        self._eval_epoch_end("test")


def build_dataloaders(config: argparse.Namespace):
    data_root = Path(config.data_root)
    train_ds = RealDESEDTrainDataset(data_root / "train")
    if config.wavmix_p > 0:
        train_ds = RealDESEDWavMixDataset(train_ds, rate=config.wavmix_p)
    val_ds = RealDESEDEvalDataset(data_root / "validation")
    test_ds = RealDESEDEvalDataset(data_root / "test")

    pw = config.num_workers > 0
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=pw,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=pw,
        collate_fn=eval_collate_identity,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=pw,
        collate_fn=eval_collate_identity,
    )
    return train_dl, val_dl, test_dl


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate BEATs on RealDESED.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"E:\workspace\audio_team\sed\models\RealDESED\data",
    )
    parser.add_argument("--run_name", type=str, default="beats_strong_frozen_bigru")
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["frozen_linear", "frozen_bigru", "last2_linear", "full"],
        default="frozen_bigru",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        choices=["scratch", "ssl", "weak", "strong"],
        default="strong",
    )
    parser.add_argument("--rnn_dim", type=int, default=256)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument("--median_window", type=int, default=9)

    # Optimizer & LR schedule
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--transformer_lr", type=float, default=None)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--lr_end", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--no_adamw", action="store_true", default=False)
    parser.add_argument("--schedule_mode", type=str, default="cos")
    parser.add_argument("--warmup_steps", type=int, default=200)

    # Augmentations
    parser.add_argument("--wavmix_p", type=float, default=0.5)
    parser.add_argument("--frame_shift_range", type=float, default=0.125)
    parser.add_argument("--mixup_p", type=float, default=0.3)
    parser.add_argument("--mixstyle_p", type=float, default=0.0)
    parser.add_argument("--filter_augment_p", type=float, default=0.3)
    parser.add_argument("--freq_warp_p", type=float, default=0.3)
    parser.add_argument("--max_time_mask_size", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    out_dir = PROJECT_ROOT / "outputs" / f"train_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl = build_dataloaders(args)
    if args.ckpt_path:
        module = RealDESEDPLModule.load_from_checkpoint(args.ckpt_path, config=args)
    else:
        module = RealDESEDPLModule(args)

    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    print(f"Run={args.run_name} | mode={args.train_mode} | trainable={trainable/1e6:.2f}M / {total/1e6:.2f}M")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:02d}",
        monitor="val/psds1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir=str(out_dir), name="logs")

    def _to_limit(val: float):
        return int(val) if val > 1.0 else val

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum,
        logger=csv_logger,
        callbacks=[ckpt_cb, lr_cb],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        limit_train_batches=_to_limit(args.limit_train_batches),
        limit_val_batches=_to_limit(args.limit_val_batches),
        limit_test_batches=_to_limit(args.limit_test_batches),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    if not args.eval_only:
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
        best_path = ckpt_cb.best_model_path or None
        test_results = trainer.test(module, dataloaders=test_dl, ckpt_path=best_path)
    else:
        test_results = trainer.test(module, dataloaders=test_dl)
    elapsed = time.perf_counter() - t0

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    )
    report = {
        "run_name": args.run_name,
        "train_mode": args.train_mode,
        "pretrained": args.pretrained,
        "trainable_params_M": round(trainable / 1e6, 3),
        "total_params_M": round(total / 1e6, 3),
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "best_checkpoint": ckpt_cb.best_model_path if not args.eval_only else args.ckpt_path,
        "test_results": test_results,
        "detailed_test_report": getattr(module, "test_report", {}),
    }
    report_path = out_dir / "test_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved test report -> {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
