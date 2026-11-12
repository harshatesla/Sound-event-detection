"""RealDESED dataset loaders for training and full-clip evaluation with PretrainedSED."""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_SED_DIR = PROJECT_ROOT / "PretrainedSED"
if str(PRETRAINED_SED_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINED_SED_DIR))

from helpers.encode import ManyHotEncoder

REALDESED_CLASSES: List[str] = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]


def _load_annotations_by_file(ann_csv: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    events_by_file: Dict[str, List[Tuple[float, float, str]]] = {}
    with ann_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            events_by_file.setdefault(fn, []).append(
                (float(row["onset"]), float(row["offset"]), row["class"])
            )
    return events_by_file


def _resample_if_needed(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    if orig_sr not in resamplers:
        resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resamplers[orig_sr](waveform)


def _pad_or_trim_1d(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    if waveform.numel() < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform[:target_samples]


class RealDESEDTrainDataset(Dataset):
    """Random 10-second crop dataset for training PretrainedSED models on RealDESED."""

    def __init__(
        self,
        split_dir: Path,
        sample_rate: int = 16_000,
        crop_duration: float = 10.0,
        classes: Optional[List[str]] = None,
    ):
        self.split_dir = Path(split_dir)
        self.audio_dir = self.split_dir / "audio"
        self.sample_rate = sample_rate
        self.crop_duration = crop_duration
        self.crop_samples = int(round(crop_duration * sample_rate))
        self.classes = classes or REALDESED_CLASSES
        self.encoder = ManyHotEncoder(
            self.classes, audio_len=crop_duration, frame_hop=160, net_pooling=4, fs=sample_rate
        )
        self.events_by_file = _load_annotations_by_file(self.split_dir / "annotations.csv")

        self.items = []
        for wav_path in sorted(self.audio_dir.glob("*.wav")):
            info = sf.info(str(wav_path))
            self.items.append(
                {
                    "filename": wav_path.name,
                    "path": wav_path,
                    "sr": info.samplerate,
                    "frames": info.frames,
                    "duration": info.frames / info.samplerate,
                }
            )
        assert len(self.items) > 0, f"No WAV files found in {self.audio_dir}"
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _encode_crop_events(
        self, events: List[Tuple[float, float, str]], crop_start: float
    ) -> torch.Tensor:
        crop_end = crop_start + self.crop_duration
        rows = []
        for onset, offset, label in events:
            if offset <= crop_start or onset >= crop_end:
                continue
            rows.append(
                {
                    "onset": max(0.0, onset - crop_start),
                    "offset": min(self.crop_duration, offset - crop_start),
                    "event_label": label,
                }
            )
        if not rows:
            return torch.zeros((len(self.classes), self.encoder.n_frames), dtype=torch.float32)
        df = pd.DataFrame(rows)
        encoded = self.encoder.encode_strong_df(df).T
        return torch.from_numpy(encoded).float()

    def __getitem__(self, idx: int):
        item = self.items[idx]
        sr = item["sr"]
        total_frames = item["frames"]
        duration = item["duration"]

        if duration > self.crop_duration:
            max_start = duration - self.crop_duration
            crop_start = float(torch.empty(1).uniform_(0.0, max_start).item())
        else:
            crop_start = 0.0

        start_frame = int(crop_start * sr)
        frames_to_read = min(total_frames - start_frame, int(np.ceil(self.crop_duration * sr)))

        with sf.SoundFile(str(item["path"])) as f:
            f.seek(start_frame)
            audio_np = f.read(frames_to_read, dtype="float32", always_2d=False)

        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        waveform = torch.from_numpy(audio_np)
        waveform = _resample_if_needed(waveform, sr, self.sample_rate, self.resamplers)
        waveform = _pad_or_trim_1d(waveform, self.crop_samples)

        events = self.events_by_file.get(item["filename"], [])
        target = self._encode_crop_events(events, crop_start)
        return {
            "audio": waveform,
            "strong": target,
            "filename": item["filename"],
        }


class RealDESEDWavMixDataset(Dataset):
    """Waveform mixup wrapper matching PretrainedSED's MixupDataset."""

    def __init__(self, dataset: Dataset, beta: float = 0.2, rate: float = 0.5):
        self.dataset = dataset
        self.beta = beta
        self.rate = rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample1 = self.dataset[idx]
        if self.rate <= 0 or torch.rand(1).item() >= self.rate:
            return sample1
        idx2 = int(torch.randint(len(self.dataset), (1,)).item())
        sample2 = self.dataset[idx2]
        lam = float(np.random.beta(self.beta, self.beta))
        lam = max(lam, 1.0 - lam)
        x1 = sample1["audio"] - sample1["audio"].mean()
        x2 = sample2["audio"] - sample2["audio"].mean()
        x = x1 * lam + x2 * (1.0 - lam)
        x = x - x.mean()
        sample1["audio"] = x
        sample1["strong"] = sample1["strong"] * lam + sample2["strong"] * (1.0 - lam)
        return sample1


class RealDESEDEvalDataset(Dataset):
    """Full-clip evaluation dataset for validation/test splits of RealDESED."""

    def __init__(
        self,
        split_dir: Path,
        sample_rate: int = 16_000,
        classes: Optional[List[str]] = None,
    ):
        self.split_dir = Path(split_dir)
        self.audio_dir = self.split_dir / "audio"
        self.sample_rate = sample_rate
        self.classes = classes or REALDESED_CLASSES
        self.events_by_file = _load_annotations_by_file(self.split_dir / "annotations.csv")
        self.wav_files = sorted(self.audio_dir.glob("*.wav"))
        assert len(self.wav_files) > 0, f"No WAV files found in {self.audio_dir}"
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int):
        wav_path = self.wav_files[idx]
        audio_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        waveform = torch.from_numpy(audio_np)
        waveform = _resample_if_needed(waveform, sr, self.sample_rate, self.resamplers)
        audio_len = waveform.numel() / self.sample_rate
        events = self.events_by_file.get(wav_path.name, [])
        return {
            "audio": waveform,
            "audio_len": audio_len,
            "filename": wav_path.name,
            "events": events,
        }


def eval_collate_identity(batch: List[dict]) -> List[dict]:
    """Keep variable-length full clips unpadded in a list for chunked evaluation."""
    return batch

