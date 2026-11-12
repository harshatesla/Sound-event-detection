# Command Log & Reasoning (`beats-strong` / `PretrainedSED`)

This directory records every command executed while reproducing `E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED`, along with the reasoning behind each command and observed results.

> **Constraints observed:**
> - All work, environments, logs, and new scripts live inside `E:\workspace\audio_team\sed\models\beats-strong`.
> - Original files inside `PretrainedSED` and dataset files in `E:\workspace\audio_team\sed\models\RealDESED\data` are never modified or deleted.

---

## Phase 1: Repository & Environment Inspection

### 1. Inspect project directory layout
**Thinking:** Before setting up an environment or running code, we need to see the exact repository layout inside `beats-strong` (scripts, configs, model folders, test audio files).
```powershell
Get-ChildItem -Force "E:\workspace\audio_team\sed\models\beats-strong" | Format-Table Name, Mode, Length, LastWriteTime; Get-ChildItem -Recurse -Depth 2 "E:\workspace\audio_team\sed\models\beats-strong" | Select-Object FullName
```
**Result:** Found `PretrainedSED/` containing `README.md`, `requirements.txt`, `config.py`, `inference.py`, `ex_audioset_strong.py`, `ex_dcase2016task2.py`, `data_util/`, `helpers/`, `models/` (`beats`, `atstframe`, `frame_passt`, `m2d`, `asit`, `frame_mn`), `resources/`, and `test_files/`.

### 2. Check available Python, uv, GPU, and `RealDESED\data` top-level structure
**Thinking:** Verify what Python versions, package managers (`uv`), and GPU hardware/CUDA driver are available on the machine, and confirm the top-level directory layout of `E:\workspace\audio_team\sed\models\RealDESED\data`. (An initial combined command including `conda --version` and recursive listing hung/was cancelled cleanly, so we checked fast direct commands.)
```powershell
uv python list --only-installed; nvidia-smi; Get-ChildItem "E:\workspace\audio_team\sed\models\RealDESED\data" | Select-Object Name, Mode
```
**Result:**
- GPU: NVIDIA GeForce GTX 1650 (4096 MiB VRAM), Driver 610.62, CUDA UMD 13.3. (Important: 4 GB VRAM means small batch sizes and FP16/mixed precision will be important.)
- Available Python matching `README.md` (`python=3.9`): `F:\Python 3.9.13\Installation\python.exe` (CPython 3.9.13).
- `uv 0.11.20` is installed.
- `RealDESED\data` contains `train`, `validation`, `test`, `annotator_quality`, `downloads`.

### 3. Inspect `PretrainedSED` Python imports and `RealDESED\data` contents
**Thinking:**
1. `PretrainedSED\requirements.txt` often omits transitive/extra imports in ML research repos (e.g. `dcase_util`, `sed_eval`, `soundfile`, `einops`, `scikit-learn`), so we scan all `.py` files for `import`/`from` lines to make sure we don't miss dependencies.
2. Inspect `RealDESED\data` split sizes, audio formats (sample rate, channels, duration), and annotation classes so we know how to run inference and evaluate training feasibility against `PretrainedSED`.
3. Create `E:\workspace\audio_team\sed\models\beats-strong\commands` to store this log.

```powershell
Get-ChildItem "E:\workspace\audio_team\sed\models\RealDESED\data" -Recurse -Depth 2 | Select-Object FullName | Out-String -Width 200; New-Item -ItemType Directory -Force "E:\workspace\audio_team\sed\models\beats-strong\commands" | Out-Null; Select-String -Path "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED\*.py", "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED\*\*.py" -Pattern "^(import |from )" | ForEach-Object { $_.Line } | Sort-Object -Unique
```
```powershell
Select-String -Path "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED\models\*\*.py" -Pattern "^(import |from )" | ForEach-Object { $_.Line } | Sort-Object -Unique
```
```powershell
@'
import csv, os, wave
from collections import Counter

base = r'E:\workspace\audio_team\sed\models\RealDESED\data'
for split in ['train', 'validation', 'test']:
    ann_path = os.path.join(base, split, 'annotations.csv')
    audio_dir = os.path.join(base, split, 'audio')
    wavs = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    with open(ann_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    classes = Counter(r['class'] for r in rows)
    srs = set()
    durs = []
    for w in wavs[:30]:
        with wave.open(os.path.join(audio_dir, w), 'rb') as wf:
            srs.add((wf.getframerate(), wf.getnchannels(), wf.getsampwidth()))
            durs.append(wf.getnframes() / wf.getframerate())
    uf = len(set(r['filename'] for r in rows))
    print(f'=== {split} ===')
    print(f'wav count: {len(wavs)}, annotation rows: {len(rows)}, unique annotated files: {uf}')
    print(f'sample audio props (first 30): {srs}, dur min/max: {min(durs):.2f}/{max(durs):.2f}s')
    print(f'classes ({len(classes)}): {dict(classes)}')
'@ | & "F:\Python 3.9.13\Installation\python.exe" -
```
**Result:**
- Identified extra imports not listed in `requirements.txt`: `dcase_util`, `sed_eval`, `soundfile`, `einops`, `scikit-learn`, `scipy`, `tqdm`.
- `RealDESED\data` has 3,704 train WAVs (17,710 events), 1,000 validation WAVs (4,919 events), and 1,007 test WAVs (5,028 events), mono 16-bit PCM at 44.1 kHz or 48 kHz, durations ~15.8–39.3s, across **15 domestic sound event classes**.

### 4. Check free disk space and verify Python 3.9.13 binary
**Thinking:** PyTorch CUDA wheels + checkpoints take several gigabytes, so verify disk space on `C:` and `E:` before creating the virtual environment.
```powershell
Get-PSDrive C,E,F | Select-Object Name, @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}, @{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}; & "F:\Python 3.9.13\Installation\python.exe" --version
```
**Result:** C: has 62.2 GB free, E: has 216.03 GB free; Python 3.9.13 verified.

---

## Phase 2: Virtual Environment & Dependency Installation

### 5. Create Python 3.9 virtual environment inside `beats-strong\.venv`
**Thinking:** All installations must be isolated inside a dedicated Python environment in the project directory (`PretrainedSED/README.md` recommends Python 3.9). Using `uv venv` with the installed CPython 3.9.13 binary avoids global/conda pollution.
```powershell
uv venv --python "F:\Python 3.9.13\Installation\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\.venv"
```
**Result:** Created `.venv` at `E:\workspace\audio_team\sed\models\beats-strong\.venv` using CPython 3.9.13.

### 6. Install PyTorch with CUDA 12.1 (`cu121`) support
**Thinking:** On Windows, plain `pip install torch` pulls CPU-only wheels. With GTX 1650 and CUDA UMD 13.3, specifying `--index-url https://download.pytorch.org/whl/cu121` installs GPU-accelerated PyTorch (`>=2.1` is required because `models/beats/backbone.py` uses `torch.nn.utils.parametrizations.weight_norm`).
```powershell
uv pip install --python "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
**Result:** Installed `torch==2.5.1+cu121`, `torchaudio==2.5.1+cu121`, `torchvision==0.20.1+cu121` cleanly into `.venv`. (Initially pulled `numpy==2.0.2`, which will be constrained to `<2` by `requirements.txt` in the next step.)

### 7. Inspect full `RealDESED\data` wav headers and class overlap with AudioSet Strong
**Thinking:** To run inference and plan training/fine-tuning on `RealDESED`, we need exact duration distributions, sample rates, channel counts across all 5,711 WAV files, and to see how the 15 `RealDESED` labels relate to the 447 `as_strong_train_classes` predicted by the pretrained `BEATs_strong_1` model.
```powershell
@'
import csv, os, wave, sys
sys.path.insert(0, r'E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED')
from data_util.audioset_classes import as_strong_train_classes, as_strong_eval_classes

base = r'E:\workspace\audio_team\sed\models\RealDESED\data'
all_classes = set()
for split in ['train', 'validation', 'test']:
    ann_path = os.path.join(base, split, 'annotations.csv')
    audio_dir = os.path.join(base, split, 'audio')
    wavs = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    with open(ann_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    classes = set(r['class'] for r in rows)
    all_classes.update(classes)
    srs = set()
    chs = set()
    durs = []
    for w in wavs:
        with wave.open(os.path.join(audio_dir, w), 'rb') as wf:
            srs.add(wf.getframerate())
            chs.add(wf.getnchannels())
            durs.append(wf.getnframes() / wf.getframerate())
    print(f'{split}: files={len(wavs)}, srs={sorted(srs)}, chs={sorted(chs)}, dur min={min(durs):.2f}s, max={max(durs):.2f}s, mean={sum(durs)/len(durs):.2f}s')

print('\nRealDESED classes:', sorted(all_classes))
as_set = set(as_strong_train_classes)
print('\nExact matches in AudioSet Strong:', [c for c in sorted(all_classes) if c in as_set])
print('\nRelevant AudioSet Strong classes:')
for rc in sorted(all_classes):
    tokens = [t for t in rc.split('_') if t not in ('open', 'close')]
    matches = [a for a in as_strong_train_classes if any(t in a.lower() for t in tokens)]
    print(f'  {rc}: {matches[:10]}')
'@ | & "F:\Python 3.9.13\Installation\python.exe" -
```
**Result:**
```
train: files=3704, srs=[8000, 16000, 24000, 32000, 44100, 48000], chs=[1], dur min=12.03s, max=39.87s, mean=23.80s
validation: files=1000, srs=[8000, 16000, 44100, 48000], chs=[1], dur min=12.69s, max=40.64s, mean=24.33s
test: files=1007, srs=[8000, 22050, 44100, 48000], chs=[1], dur min=12.22s, max=39.93s, mean=23.64s
```
- `RealDESED` has 15 domestic sound classes (`bell_ringing`, `coffee_machine`, `cutlery_dishes`, `door_open_close`, `footsteps`, `keyboard_typing`, `keychain`, `light_switch`, `microwave`, `phone_ringing`, `running_water`, `toilet_flushing`, `vacuum_cleaner`, `wardrobe_drawer_open_close`, `window_open_close`).
- None match AudioSet Strong class strings verbatim, though most have close AudioSet Strong equivalents (e.g. `Walk, footsteps`, `Door`, `Computer keyboard`, `Microwave oven`, `Telephone bell ringing`, `Water tap, faucet`, `Toilet flush`, `Vacuum cleaner`, `Drawer open or close`).
- Because sample rates vary (`8k`–`48k`) and durations vary (`12.03s`–`40.64s`), inference and training on `RealDESED` must resample to `16000` Hz (matching `PretrainedSED`) and handle variable durations (chunking or 10s random/fixed crops).

### 8. Install `PretrainedSED/requirements.txt` and missing code-imported packages
**Thinking:** While `PretrainedSED/requirements.txt` specifies core dependencies (`numpy<2`, `librosa`, `timm`, `nnAudio`, `datasets`, `pytorch-lightning`, `transformers`, `sed_scores_eval==0.0.3`, `intervaltree`, etc.), scanning the codebase also showed imports of `dcase_util`, `sed_eval`, `soundfile`, `einops`, and `scikit-learn`. Using `--link-mode=copy` avoids cross-drive hardlink warnings between `C:` and `E:`.
```powershell
uv pip install --link-mode=copy --python "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -r "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED\requirements.txt" dcase_util sed_eval soundfile einops scikit-learn
```
**Result:** Failed building `sed-scores-eval==0.0.3`: `ModuleNotFoundError: No module named 'numpy'` inside `uv`'s isolated build environment because `sed-scores-eval==0.0.3` imports `numpy` in `setup.py` without declaring it in `pyproject.toml`.

### 9. Install build prerequisites (`numpy<2`, `setuptools`, `wheel`, `cython`) and retry with `--no-build-isolation`
**Thinking:** Installing `numpy<2`, `setuptools`, `wheel`, and `cython` into `.venv` first and then passing `--no-build-isolation` lets `sed-scores-eval==0.0.3` (and any other legacy `setup.py` packages) resolve `numpy` at build time. (Also note PowerShell 5.1 requires `; if ($LASTEXITCODE -eq 0)` instead of `&&`, after an initial `&&` syntax error.)
```powershell
uv pip install --link-mode=copy --python "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "numpy<2" setuptools wheel cython; if ($LASTEXITCODE -eq 0) { uv pip install --link-mode=copy --no-build-isolation --python "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -r "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED\requirements.txt" dcase_util sed_eval soundfile einops scikit-learn }
```
**Result:** Successfully installed all dependencies (`numpy==1.26.4`, `sed-scores-eval==0.0.3`, `dcase-util==0.2.20`, `sed-eval==0.2.1`, `pytorch-lightning==2.6.0`, `librosa==0.11.0`, `timm==1.0.30`, `transformers==4.57.6`, `nnaudio==0.3.4`, `soundfile==0.13.1`, `einops==0.8.2`, `scikit-learn==1.6.1`, etc.) into `.venv`.

---

## Phase 3: Step 1 — Verify CUDA & Run Repository Inference

### 10. Verify PyTorch CUDA availability and run built-in `inference.py` on the bundled example WAV
**Thinking:** Confirm `torch.cuda` detects the GTX 1650 properly, then run `PretrainedSED/inference.py` with `BEATs` (the repo's primary `BEATs_strong_1` model) on the included test audio file to verify checkpoint auto-download, preprocessing, model forward pass, and decoding. Working directory must be `PretrainedSED` because `config.py` uses relative `"resources"`.
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Push-Location "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED"
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" inference.py --cuda --model_name="BEATs" --audio_file="test_files/752547__iscence__milan_metro_coming_in_station.wav"
Pop-Location
```
**Result:**
- `cuda: True NVIDIA GeForce GTX 1650` confirmed.
- `inference.py` failed on import: `dcase_util/utils/utils.py` imports `pkg_resources`, which was removed in `setuptools==82.0.1`. (PowerShell returned exit 0 only because `Pop-Location` succeeded after `python.exe`.)

### 11. Downgrade `setuptools<81` so `pkg_resources` is available for `dcase_util`
**Thinking:** `setuptools>=82` removed `pkg_resources`. Pinning `setuptools<81` restores `pkg_resources` cleanly inside `.venv` without altering any repo source files.
```powershell
uv pip install --link-mode=copy --python "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "setuptools<81"
```
**Result:** Installed `setuptools==80.10.2`.

### 12. Re-run `PretrainedSED/inference.py` with `BEATs` on CUDA
**Thinking:** With `pkg_resources` restored, re-run `inference.py --cuda --model_name="BEATs"` on the bundled test WAV (`test_files/752547__iscence__milan_metro_coming_in_station.wav`), propagating Python's exit code properly.
```powershell
Push-Location "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED"
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" inference.py --cuda --model_name="BEATs" --audio_file="test_files/752547__iscence__milan_metro_coming_in_station.wav"
$ec = $LASTEXITCODE
Pop-Location
exit $ec
```
**Result:** Succeeded (exit 0). Downloaded `PretrainedSED/resources/BEATs_strong_1.pt` (347 MB) and detected `Train`, `Subway, metro, underground`, `Train wheels squealing`, `Mechanisms`, `Background noise`, etc. across thresholds `0.1`, `0.2`, `0.5`.

### 13. Run `PretrainedSED/inference.py` directly on `RealDESED` test audio files
**Thinking:** Verify that the unmodified `inference.py` works on `E:\workspace\audio_team\sed\models\RealDESED\data\test\audio` files (which have varying sample rates like 44.1/48 kHz and durations ~20–35s) and compare detected AudioSet Strong labels against the known ground-truth events in `test/annotations.csv` (`000020.wav`: `cutlery_dishes`, `footsteps`; `000023.wav`: `running_water`, `keychain`).
```powershell
Push-Location "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED"
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" inference.py --cuda --model_name="BEATs" --audio_file="E:\workspace\audio_team\sed\models\RealDESED\data\test\audio\000020.wav"
if ($LASTEXITCODE -eq 0) {
  & "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" inference.py --cuda --model_name="BEATs" --audio_file="E:\workspace\audio_team\sed\models\RealDESED\data\test\audio\000023.wav"
}
$ec = $LASTEXITCODE
Pop-Location
exit $ec
```
**Result:** Succeeded (exit 0) for both `RealDESED` test files:
- `000020.wav` (31.57s): detected `Walk, footsteps` (6.00–9.80s, 10.00–19.56s, 27.84–29.80s), `Generic impact sounds` / `Mechanisms` / `Coin (dropping)` (0.32–6.68s, 20.40–27.68s), closely matching `RealDESED` ground truth (`footsteps` and `cutlery_dishes`).
- `000023.wav` (34.47s): detected `Water tap, faucet` / `Sink (filling or washing)` / `Pour` (0.60–8.12s, 9.76–11.56s, 12.80–16.68s, 31.12–34.12s) and `Generic impact sounds` / `Coin (dropping)` / `Rattle` (16.88–30.00s), closely matching `RealDESED` ground truth (`running_water` and `keychain`).

### 14. Run batch inference smoke test on `RealDESED\data\test\audio` (first 20 files)
**Thinking:** Created `scripts/infer_realdesed.py` (without modifying `PretrainedSED`) to batch-process `RealDESED` WAVs at 16 kHz in 10s chunks using `BEATs_strong_1`, trim padded frames to the exact audio duration, measure throughput and peak VRAM on GTX 1650, and save event-level CSVs. Run a 20-file test first before running all 1,007 test files.
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\infer_realdesed.py" --cuda --max_files 20
```
**Result:** Succeeded (exit 0).
- `BEATs_strong_1` loaded with `91.00M` parameters.
- Processed 20 files (`458.37s` total audio) in `28.88s` (`15.87x` real-time) with `749.61 MB` peak VRAM.
- Saved `outputs/inference_realdesed_test/predictions_th0.2.csv` (330 events) and `predictions_th0.5.csv` (83 events); top detections include `Generic impact sounds`, `Walk, footsteps`, `Surface contact`, `Mechanisms`, `Computer keyboard`, `Microwave oven`, `Water tap, faucet`, `Toilet flush`, `Drawer open or close`, `Door`.

---

## Phase 4: Step 2 — Check Training Feasibility on GTX 1650 (4 GB VRAM)

### 15. Inspect CPU/RAM and benchmark training peak VRAM across fine-tuning modes
**Thinking:** To determine whether training/fine-tuning `BEATs_strong_1` on `RealDESED\data` is feasible on this machine's GTX 1650 (4 GB VRAM), inspect logical CPU cores and system RAM and run `scripts/check_training_feasibility.py` measuring `forward + backward + AdamW.step()` peak VRAM in FP16 mixed precision across `frozen_linear`, `frozen_bigru256`, `last2_linear`, and `full_finetune` at varying batch sizes.
```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object NumberOfLogicalProcessors, @{N='TotalRAM_GB';E={[math]::Round($_.TotalPhysicalMemory/1GB,2)}}
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\check_training_feasibility.py"
```
**Result:** Succeeded (exit 0).
- System: 8 logical processors, `15.88 GB` RAM; GPU free/total VRAM: `3294.8 MB / 4095.7 MB`.
- Every training mode fits in 4 GB VRAM with FP16 mixed precision:
  - `frozen_linear` (`0.02M` trainable): `bs=8` -> `835.5 MB`, `bs=16` -> `1308.2 MB`
  - `frozen_bigru256` (`2.77M` trainable): `bs=8` -> `868.4 MB`, `bs=16` -> `1342.4 MB`
  - `last2_linear` (`14.20M` trainable): `bs=4` -> `2199.1 MB`, `bs=8` -> `4029.5 MB` (exceeds free 3294.8 MB, relies on shared memory — unsafe; use `bs=4`)
  - `full_finetune` (`90.33M` trainable): `bs=1` -> `1762.5 MB`, `bs=2` -> `2026.1 MB`, `bs=4` -> `3055.8 MB`
- Conclusion: Training on `RealDESED\data` is **fully possible** on this machine, including full end-to-end `BEATs_strong_1` fine-tuning (`bs=2` or `bs=4` with gradient accumulation) as well as fast frozen-backbone / BiGRU head training (`bs=16`).

### 16. Verify `sed_scores_eval==0.0.3` behavior and `soundfile` + `torchaudio` crop-resampling speed
**Thinking:** Before writing the full `RealDESED` training and evaluation pipeline, verify how `sed_scores_eval==0.0.3` handles empty ground-truth clips (like the 1 unannotated file in `RealDESED/validation`) and benchmark fast partial-read + resample via `soundfile.SoundFile` + `torchaudio`.
```powershell
@'
import time, os, wave
import numpy as np
import soundfile as sf
import torch, torchaudio
import sed_scores_eval
from sed_scores_eval.base_modules.scores import create_score_dataframe

# 1. Check sed_scores_eval with empty gt file
classes = ["a", "b"]
scores = {
    "f1": create_score_dataframe(np.array([[0.8, 0.1], [0.9, 0.1]]), np.array([0.0, 0.5, 1.0]), classes),
    "f2": create_score_dataframe(np.array([[0.1, 0.1], [0.1, 0.1]]), np.array([0.0, 0.5, 1.0]), classes),
}
gt = {"f1": [(0.0, 1.0, "a")], "f2": []}
durs = {"f1": 1.0, "f2": 1.0}
try:
    p = sed_scores_eval.intersection_based.psds(scores, gt, durs, dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=None, alpha_ct=0, alpha_st=1, num_jobs=1)
    print("psds with empty gt ok:", p[0], p[1])
except Exception as e:
    print("psds with empty gt failed:", repr(e))

# 2. Benchmark soundfile partial read + torchaudio resample on 200 train wavs
audio_dir = r"E:\workspace\audio_team\sed\models\RealDESED\data\train\audio"
files = sorted(os.listdir(audio_dir))[:200]
resamplers = {}
t0 = time.perf_counter()
for fn in files:
    p = os.path.join(audio_dir, fn)
    with sf.SoundFile(p) as f:
        sr = f.samplerate
        n = min(f.frames, int(10.0 * sr))
        f.seek(0)
        x = f.read(n, dtype="float32")
    t = torch.from_numpy(x)
    if sr != 16000:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
        t = resamplers[sr](t)
dt = time.perf_counter() - t0
print(f"Loaded & resampled 200 x 10s crops in {dt:.3f}s ({200/dt:.1f} files/s)")
'@ | & "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -
```
**Result:**
- `sed_scores_eval==0.0.3` works with empty-event clips (`gt["f2"] = []`) as long as all score columns have at least one ground-truth event across the evaluated split.
- `soundfile` partial seek/read + `torchaudio.transforms.Resample` processes 200 x 10s crops in `11.496s` (`17.4 files/s` single-threaded).

### 17. Smoke-test `scripts/train_realdesed.py` end-to-end (`frozen_bigru`, limited batches)
**Thinking:** Created `scripts/realdesed_dataset.py` (random 10s crop train dataset with partial WAV seeking, waveform mixup wrapper, and full-clip eval dataset) and `scripts/train_realdesed.py` (supporting `frozen_linear`, `frozen_bigru`, `last2_linear`, and `full` fine-tuning, `PretrainedSED` augmentations, `CSVLogger`, `ModelCheckpoint`, and `sed_scores_eval` PSDS1/pAUROC metrics). Run a 1-epoch mini smoke test (`10` train batches, `30` val/test batches) on Windows first to confirm DataLoader multiprocessing, FP16 training, checkpoint reload, and `sed_scores_eval` metrics work without errors.
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\train_realdesed.py" --run_name smoke_test --train_mode frozen_bigru --batch_size 8 --num_workers 2 --epochs 1 --limit_train_batches 10 --limit_val_batches 30 --limit_test_batches 30
```
**Result:** Succeeded (exit 0).
- Pipeline verified end-to-end on Windows (`frozen_bigru`, 2.77M/93.09M trainable params, `828.88 MB` peak VRAM, `test/psds1_macro=0.1168`, `test/pauroc=0.2561` on the 30-clip subset after just 10 mini-batches).
- Observed performance bottleneck: validation took ~1.1s/clip (30 clips in 33s) and total run took `256.74s` for 10 train + 30 val + 30 test batches, so profiling CPU resampling vs GPU inference vs `batched_decode_preds` is needed before full-dataset multi-epoch training.

### 18. Profile validation step bottlenecks (disk read, CPU vs GPU resampling, model forward, `batched_decode_preds`)
**Thinking:** Full validation has 1,000 clips and test has 1,007 clips; at 1.1s/clip, each full validation would take ~18 minutes. Measure each sub-step on 30 validation clips to see what is slow so we can optimize `train_realdesed.py` before running full multi-epoch training.
```powershell
@'
import os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, r'E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED')
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper
sys.path.insert(0, r'E:\workspace\audio_team\sed\models\beats-strong\scripts')
from realdesed_dataset import REALDESED_CLASSES

val_dir = Path(r'E:\workspace\audio_team\sed\models\RealDESED\data\validation\audio')
wavs = sorted(val_dir.glob('*.wav'))[:30]
device = torch.device('cuda')
os.chdir(r'E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED')
model = PredictionsWrapper(BEATsWrapper(), checkpoint='BEATs_strong_1', n_classes_strong=15).eval().to(device)

t_read = t_cpu_res = t_gpu_res = t_fwd = t_dec_5 = t_dec_0 = 0.0
resamplers_cpu = {}
resamplers_gpu = {}
for p in wavs:
    t0 = time.perf_counter()
    x, sr = sf.read(str(p), dtype='float32')
    t1 = time.perf_counter()
    t_read += t1 - t0
    w = torch.from_numpy(x)
    if sr != 16000:
        if sr not in resamplers_cpu: resamplers_cpu[sr] = torchaudio.transforms.Resample(sr, 16000)
        _ = resamplers_cpu[sr](w)
    t2 = time.perf_counter()
    t_cpu_res += t2 - t1

    wg = w.to(device)
    if sr != 16000:
        if sr not in resamplers_gpu: resamplers_gpu[sr] = torchaudio.transforms.Resample(sr, 16000).to(device)
        wg = resamplers_gpu[sr](wg)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    t_gpu_res += t3 - t2

    wlen = wg.numel()
    seg = 160000
    n_c = wlen // seg + int(wlen % seg != 0)
    chunks = [torch.nn.functional.pad(wg[i*seg:(i+1)*seg], (0, max(0, seg - len(wg[i*seg:(i+1)*seg])))) for i in range(n_c)]
    cb = torch.stack(chunks, dim=0)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        y, _ = model(model.mel_forward(cb))
    y = y.transpose(0, 1).reshape(1, 15, -1)
    tf = max(1, min(y.shape[2], int(round((wlen / 16000) * 25))))
    probs = torch.sigmoid(y[:, :, :tf]).float()
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    t_fwd += t4 - t3

    enc = ManyHotEncoder(REALDESED_CLASSES, audio_len=tf / 25.0)
    batched_decode_preds(probs, [p.name], enc, thresholds=(0.5,), median_filter=9)
    t5 = time.perf_counter()
    t_dec_5 += t5 - t4

    batched_decode_preds(probs, [p.name], enc, thresholds=(), median_filter=9)
    t6 = time.perf_counter()
    t_dec_0 += t6 - t5

print(f'30 clips: read={t_read:.2f}s, cpu_res={t_cpu_res:.2f}s, gpu_res={t_gpu_res:.2f}s, fwd={t_fwd:.2f}s, dec(0.5)={t_dec_5:.2f}s, dec(())={t_dec_0:.2f}s')
'@ | & "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -
```
**Result:**
`30 clips: read=1.61s, cpu_res=0.28s, gpu_res=0.90s, fwd=31.62s, dec(0.5)=0.13s, dec(())=0.05s`
- Disk read, resampling, and decoding together take `<2s` for 30 clips; `94%` of the time (`31.62s`) was `BEATs` forward pass under FP16 AMP.

### 19. Profile `mel_forward` vs. `BEATs` forward pass under FP16 (`amp=True`) vs. FP32 (`amp=False`)
**Thinking:** GTX 1650 (Turing TU117) lacks FP16 Tensor Cores, so FP16 GEMMs/attention can actually run slower than FP32. Measure exact `mel_forward` and `BEATs` forward times at `bs=16` for both FP16 AMP and FP32.
```powershell
@'
import os, sys, time, torch
sys.path.insert(0, r'E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED')
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper

os.chdir(r'E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED')
device = torch.device('cuda')
model = PredictionsWrapper(BEATsWrapper(), checkpoint='BEATs_strong_1', n_classes_strong=15).eval().to(device)

for amp in [True, False]:
    x = torch.randn(16, 160_000, device=device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
        _ = model(model.mel_forward(x))
    torch.cuda.synchronize()
    t_mel = t_fwd = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
            mel = model.mel_forward(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
            y, _ = model(mel)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_mel += t1 - t0
        t_fwd += t2 - t1
    print(f'amp={str(amp):5s} bs=16: mel={t_mel/3:.3f}s ({t_mel/48*1000:.1f}ms/clip), fwd={t_fwd/3:.3f}s ({t_fwd/48*1000:.1f}ms/clip)')
'@ | & "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -
```
**Result:**
- `amp=True  bs=16: mel=0.027s (1.7ms/clip), fwd=4.448s (278.0ms/clip)`
- `amp=False bs=16: mel=0.028s (1.7ms/clip), fwd=1.580s (98.7ms/clip)`
- **Key finding:** FP32 is **2.82x faster** than FP16 AMP on GTX 1650 (because TU117 has no Tensor Cores).

### 20. Benchmark FP32 training peak VRAM and step throughput across training modes
**Thinking:** Since FP32 is ~2.8x faster than FP16 on GTX 1650, check whether `frozen_linear`, `frozen_bigru256`, `last2_linear`, and `full_finetune` fit in 3.3 GB free VRAM in FP32 (`32-true`) and measure exact training step speed.
```powershell
@'
import gc, os, sys, time, torch, torch.nn as nn
sys.path.insert(0, r'E:\workspace\audio_team\sed\models\beats-strong\scripts')
from check_training_feasibility import build_model

device = torch.device('cuda')
for mode, bs in [('frozen_linear', 16), ('frozen_bigru256', 16), ('frozen_bigru256', 8), ('last2_linear', 4), ('full_finetune', 2), ('full_finetune', 4)]:
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats(device)
    try:
        m = build_model(mode, device).train()
        if mode.startswith('frozen'):
            m.model.eval()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        x = torch.randn(bs, 160_000, device=device)
        y = torch.zeros(bs, 15, 250, device=device)
        for step in range(3):
            if step == 1:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            mel = m.mel_forward(x)
            yhat, _ = m(mel)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 2
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f'[PASS FP32] {mode:15s} bs={bs:2d}: {dt*1000:6.1f}ms/step ({dt/bs*1000:5.1f}ms/sample) | peak={peak:7.1f} MB')
    except RuntimeError as e:
        print(f'[FAIL FP32] {mode:15s} bs={bs:2d}: {str(e).splitlines()[0]}')
    finally:
        for k in ('m', 'opt', 'x', 'y', 'mel', 'yhat', 'loss'):
            if k in locals(): del locals()[k]
        torch.cuda.empty_cache(); gc.collect()
'@ | & "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" -
```
**Result:**
- `frozen_linear   bs=16`: `1706.8ms/step` (`106.7ms/sample`) | peak VRAM `1267.9 MB`
- `frozen_bigru256 bs=16`: `1813.6ms/step` (`113.4ms/sample`) | peak VRAM `1302.1 MB`
- `frozen_bigru256 bs= 8`: `911.4ms/step` (`113.9ms/sample`) | peak VRAM `848.0 MB`
- `last2_linear    bs= 4`: `902.9ms/step` (`225.7ms/sample`) | peak VRAM `2590.4 MB`
- `full_finetune   bs= 2`: `508.5ms/step` (`254.3ms/sample`) | peak VRAM `2287.5 MB`
- `full_finetune   bs= 4`: `1064.2ms/step` (`266.1ms/sample`) | peak VRAM `3637.8 MB` (spills slightly beyond `3294.8 MB` dedicated free VRAM; `bs=2` fits cleanly with `>1 GB` headroom)
- Conclusion: Running in FP32 (`32-true`) is **2.82x faster** than FP16 AMP on GTX 1650 while fitting `frozen_linear`, `frozen_bigru256`, `last2_linear`, and `full_finetune` (`bs=2` with gradient accumulation) within dedicated VRAM.

### 21. Benchmark `infer_realdesed.py` with FP32 (`--no_amp`) on 50 `RealDESED` test files
**Thinking:** Since FP32 is 2.82x faster than FP16 on GTX 1650, verify throughput of `scripts/infer_realdesed.py --cuda --no_amp` on 50 test files to decide if running all 1,007 test files is practical in a few minutes.
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\infer_realdesed.py" --cuda --no_amp --max_files 50
```
**Result:** Succeeded (exit 0).
- 50 files (`1178.24s` total audio) finished in `24.3s` (`2.06 files/s`, **`48.49x` real-time**, **`589.6 MB` peak VRAM**) — 3x faster and lower memory than FP16 AMP.

### 22. Run full-split Step 1 inference on all 1,007 `RealDESED\data\test` files
**Thinking:** At `48.49x` real-time (`~2.06 files/s`), running `BEATs_strong_1` inference across the entire 1,007-file `RealDESED\data\test` split takes only ~8 minutes and produces complete event predictions (`predictions_th0.2.csv`, `predictions_th0.5.csv`) across the test dataset.
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\infer_realdesed.py" --cuda --no_amp
```
**Result:** Succeeded (exit 0).
- Processed all **1,007** test clips (`23,807.31s` = 6.61 hours of audio) in `386.93s` (`6.45 min`), achieving **`61.53x` real-time** (`2.60 files/s`) and **`589.6 MB` peak VRAM**.
- Saved `outputs/inference_realdesed_test/predictions_th0.2.csv` (`17,516` events) and `predictions_th0.5.csv` (`4,782` events).
- Top detections at `th=0.5`: `Generic impact sounds` (1145), `Background noise` (910), `Mechanisms` (829), `Computer keyboard` (455), `Walk, footsteps` (214), `Water tap, faucet` (197), `Surface contact` (155), `Toilet flush` (135), `Microwave oven` (129), `Ringtone` (76), `Vacuum cleaner` (46).

---

## Phase 5: Step 2 — Full Training & Evaluation on `RealDESED`

### 23. Train `BEATs_strong_1` + trainable 2-layer BiGRU (`frozen_bigru`) on full `RealDESED` (`train` -> `validation` -> `test`)
**Thinking:** Run a full 3-epoch training job over all 3,704 `RealDESED/train` clips (`bs=16`, FP32 `32-true`, `max_lr=1e-3`, AdamW + cosine decay, `PretrainedSED` augmentations), validate on all 1,000 `RealDESED/validation` clips, and evaluate the best checkpoint on all 1,007 `RealDESED/test` clips (`PSDS1`, `PSDS2`, `pAUROC`, per-class PSDS1/PSDS2, and decoded test CSVs).
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\train_realdesed.py" --run_name beats_strong_frozen_bigru_e3 --train_mode frozen_bigru --pretrained strong --batch_size 16 --num_workers 4 --epochs 3 --check_val_every_n_epoch 3 --precision 32-true --max_lr 1e-3 --wavmix_p 0.3 --mixup_p 0.2 --filter_augment_p 0.2 --freq_warp_p 0.2 --max_time_mask_size 0.1
```
**Result:** Succeeded (exit 0).
- Duration: `2289.99s` (`38.17 min`) for 3 full training epochs (`3,704` clips/epoch), full validation (`1,000` clips), and full test (`1,007` clips); peak VRAM **`1320.94 MB`**.
- Best checkpoint saved at `outputs/train_beats_strong_frozen_bigru_e3/checkpoints/best-epoch=02.ckpt`.
- Held-out **Test metrics** (`RealDESED/data/test`, all 1,007 clips):
  - `test/psds1`: **`0.3438`** (`test/psds1_macro`: **`0.5511`**)
  - `test/psds2`: **`0.5745`** (`test/psds2_macro`: **`0.7598`**)
  - `test/pauroc`: **`0.8235`**
- Strongest per-class PSDS1 / PSDS2: `vacuum_cleaner` (`0.920 / 0.988`), `running_water` (`0.814 / 0.949`), `keyboard_typing` (`0.787 / 0.950`), `coffee_machine` (`0.686 / 0.810`), `toilet_flushing` (`0.635 / 0.971`), `phone_ringing` (`0.619 / 0.764`), `bell_ringing` (`0.616 / 0.777`), `microwave` (`0.615 / 0.876`).
- Saved test report and thresholded predictions in `outputs/train_beats_strong_frozen_bigru_e3/`.

### 24. Run full end-to-end fine-tuning (`--train_mode full`) for 1 epoch on `RealDESED`
**Thinking:** To empirically verify that full end-to-end gradient-based fine-tuning of all `90.33M` parameters of `BEATs_strong_1` (with `lr_decay=0.95` via `separate_params()`) also succeeds on `RealDESED` within the GTX 1650's 4 GB VRAM, run 1 full epoch (`bs=2`, `grad_accum=8`, effective `bs=16`, `max_lr=1e-4`, FP32 `32-true`) followed by full `validation` and `test`.
```powershell
& "E:\workspace\audio_team\sed\models\beats-strong\.venv\Scripts\python.exe" "E:\workspace\audio_team\sed\models\beats-strong\scripts\train_realdesed.py" --run_name beats_strong_full_e1 --train_mode full --pretrained strong --batch_size 2 --grad_accum 8 --num_workers 4 --epochs 1 --check_val_every_n_epoch 1 --precision 32-true --max_lr 1e-4 --lr_decay 0.95 --wavmix_p 0.3 --mixup_p 0.2 --filter_augment_p 0.2 --freq_warp_p 0.2 --max_time_mask_size 0.1
```
**Result:** Succeeded (exit 0).
- Duration: `1826.76s` (`30.45 min`) for 1 full training epoch (`1,852` micro-batches / `232` optimizer steps), full validation (`1,000` clips), and full test (`1,007` clips).
- Trainable parameters: **`90.323M / 90.335M`** (full end-to-end `BEATs_strong_1` + 15-class head); peak VRAM: **`2749.74 MB`** (within free 3294.8 MB VRAM).
- Best checkpoint saved at `outputs/train_beats_strong_full_e1/checkpoints/best-epoch=00.ckpt`.
- Held-out **Test metrics** (`RealDESED/data/test`, all 1,007 clips):
  - `test/psds1`: **`0.0854`** (`test/psds1_macro`: **`0.3982`**)
  - `test/psds2`: **`0.1446`** (`test/psds2_macro`: **`0.5243`**)
  - `test/pauroc`: **`0.5627`**
- Strongest per-class PSDS1 after 1 epoch: `vacuum_cleaner` (`0.913`), `running_water` (`0.822`), `keyboard_typing` (`0.784`), `toilet_flushing` (`0.647`), `phone_ringing` (`0.598`), `microwave` (`0.556`).

### 25. Verify repository cleanliness and confirm original `PretrainedSED` files were not modified or deleted
**Thinking:** Verify via `git status` inside `PretrainedSED` (if a git repo) and list newly created files inside `E:\workspace\audio_team\sed\models\beats-strong` to confirm strict compliance with the constraint that no original contents were modified or deleted.
```powershell
git -C "E:\workspace\audio_team\sed\models\beats-strong\PretrainedSED" status -s; Get-ChildItem "E:\workspace\audio_team\sed\models\beats-strong" -Force | Select-Object Name, Mode
```
**Result:** Succeeded (exit 0).
- `git -C PretrainedSED status -s` shows only `?? resources/BEATs_strong_1.pt` (the auto-downloaded checkpoint inside `resources/`, matching `resources/README.md`). Zero original files were modified or deleted.
- New artifacts inside `E:\workspace\audio_team\sed\models\beats-strong`:
  - `.venv\` (isolated Python 3.9.13 environment with all packages)
  - `commands\README.md` (complete command and reasoning log)
  - `scripts\` (`infer_realdesed.py`, `check_training_feasibility.py`, `realdesed_dataset.py`, `train_realdesed.py`)
  - `outputs\` (`inference_realdesed_test/`, `train_smoke_test/`, `train_beats_strong_frozen_bigru_e3/`, `train_beats_strong_full_e1/`)
