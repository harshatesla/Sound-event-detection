# Reproducing the RealDESED Environment on Another Machine

This guide provides step-by-step instructions to recreate the exact Python environment, dependencies, dataset, pretrained weights, patches, and verification checks used for **RealDESED** (`E:\workspace\audio_team\sed\models\RealDESED`) on a new machine.

---

## 1. Tested Environment & System Requirements

| Component | Tested Configuration | Notes |
|---|---|---|
| **OS** | Windows 10 / 11 (PowerShell) | Linux / macOS steps are also included below |
| **Python** | `3.10.13` | Python 3.10 required for compatibility with all pinned wheels |
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB VRAM) | Any CUDA-capable NVIDIA GPU works; $\ge 11$ GB allows full `--batch_size 48` |
| **PyTorch / CUDA** | `torch==2.5.1+cu121`, `torchaudio==2.5.1+cu121` | Explicitly installed from the CUDA 12.1 PyTorch wheel index |
| **Transformers** | `transformers==4.57.6` (`<5.0`) | **Critical pin** (see Troubleshooting) |
| **Setuptools** | `setuptools==65.5.0` (`<81`) | Required for `pkg_resources` used by `dcase_util` / `sed_eval` |
| **CLI Tools** | `curl` (`curl.exe` on Windows), `ffmpeg` (`9.0.2` tested) | Needed for reliable Zenodo downloads and audio decoding |
| **Disk Space** | ~20 GB free | ~8.12 GB archives + ~8 GB extracted audio + ~329 MB model + ~3 GB packages |

---

## 2. Step-by-Step Setup (Windows PowerShell)

> Execute all commands from inside the repository root directory (e.g., `RealDESED\`).

### Step 1: Clone / Copy the Repository
If cloning fresh from upstream instead of copying this folder directly:
```powershell
git clone https://github.com/fschmid56/RealDESED.git
cd RealDESED
```

### Step 2: Install FFmpeg (if missing)
Verify `ffmpeg` is available on `PATH`:
```powershell
ffmpeg -version
```
If not found on Windows, install via WinGet and restart your terminal:
```powershell
winget install Gyan.FFmpeg
```
*(On Linux/Ubuntu: `sudo apt update && sudo apt install -y ffmpeg`)*

### Step 3: Create an Isolated Python 3.10 Virtual Environment
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
*(If PowerShell blocks script activation, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or bypass activation by calling `.\.venv\Scripts\python.exe` directly.)*

### Step 4: Install PyTorch + Torchaudio with CUDA 12.1
Install CUDA-enabled PyTorch wheels **first** from the PyTorch wheel index (default PyPI wheels on Windows install CPU-only builds):
```powershell
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
*(If your machine has a newer GPU/driver requiring CUDA 12.4, use `--index-url https://download.pytorch.org/whl/cu124`. If CPU-only, omit `--index-url`, though training/inference will be very slow.)*

### Step 5: Install Pinned Dependencies
Install the verified package versions from our reproduction environment:
```powershell
pip install `
  "setuptools==65.5.0" `
  "transformers==4.57.6" `
  "einops==0.8.2" `
  "numpy==2.2.6" `
  "pandas==2.3.3" `
  "pytorch-lightning==2.6.6" `
  "sed_scores_eval==0.0.4" `
  "sed_eval==0.2.1" `
  "torchcodec==0.16.0" `
  "wandb==0.31.0" `
  "librosa==0.11.0" `
  "soundfile==0.14.0" `
  "scipy==1.15.3" `
  "scikit-learn==1.7.2" `
  "matplotlib==3.10.9"
```

<details>
<summary>Full exact <code>pip freeze</code> snapshot from the reference <code>.venv</code> (click to expand)</summary>

> If installing all lines below into a fresh environment, use `--extra-index-url https://download.pytorch.org/whl/cu121` so `torch==2.5.1+cu121` and `torchaudio==2.5.1+cu121` resolve properly.

```text
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.16.0
async-timeout==5.0.1
attrs==26.1.0
audioread==3.1.0
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.2
cwevals==0.0.1b1
cycler==0.12.1
Cython==3.3.1
dcase_util==0.2.20
decorator==5.3.1
distro==1.9.0
einops==0.8.2
exceptiongroup==1.3.1
filelock==4.1.0
fonttools==4.65.0
frozenlist==1.8.0
fsspec==2026.9.0
future==1.0.0
googleapis-common-protos==1.76.0
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpcore2==2.14.0
httpx==0.28.1
httpx2==2.14.0
huggingface_hub==0.36.2
idna==3.20
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
lazy-dataset==0.0.15
lazy_loader==0.6
librosa==0.11.0
lightning-utilities==0.15.3
llvmlite==0.50.0
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.4
multidict==7.1.0
networkx==3.4.2
numba==0.68.1
numpy==2.2.6
opentelemetry-api==1.45.2
opentelemetry-exporter-http-transport==0.66b2
opentelemetry-exporter-otlp-common==0.66b2
opentelemetry-exporter-otlp-proto-common==1.45.2
opentelemetry-exporter-otlp-proto-http==1.45.2
opentelemetry-proto==1.45.2
opentelemetry-sdk==1.45.2
opentelemetry-semantic-conventions==0.66b2
packaging==26.3
pandas==2.3.3
pathlib==1.0.1
pillow==12.3.0
platformdirs==4.12.4
pooch==1.9.0
propcache==0.5.4
protobuf==7.36.2
pycparser==3.11
pydantic==2.14.3
pydantic_core==2.50.4
pydot-ng==2.0.0
Pygments==2.21.0
pyparsing==3.3.3
python-dateutil==3.0.0
python-magic==0.4.27
pytorch-lightning==2.6.6
pytz==2026.6
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.7.2
scipy==1.15.3
sed_eval==0.2.1
sed_scores_eval==0.0.4
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
soundfile==0.14.0
soxr==1.1.0
sympy==1.13.1
threadpoolctl==3.8.0
tokenizers==0.22.2
tomli==2.5.0
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchcodec==0.16.0
torchmetrics==1.9.0
tqdm==4.70.1
transformers==4.57.6
truststore==0.10.5
typer==0.27.3
typing-inspection==0.4.5
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
validators==0.35.0
wandb==0.31.0
xxhash==4.0.2
yarl==1.25.1
```

</details>

### Step 6: Verify Python Environment & Unit Tests
Check PyTorch CUDA detection and Transformers compatibility:
```powershell
python -c "import torch, torchaudio, transformers; from transformers import get_cosine_schedule_with_warmup; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'Transformers:', transformers.__version__)"
```
Run the repository's 24 component tests (this works without downloading the dataset):
```powershell
python agent_tests\test.py
```
Expected result: `All component tests passed.` (`test_dataset.py`: 10 tests, `test_train.py`: 3 tests, `test_utils.py`: 11 tests).

---

## 3. Code Modification Checklist (If Cloning Fresh from Upstream)

If you cloned from the original GitHub repository rather than copying this modified directory, apply the checkpoint fallback patch in `train.py` (around lines 417–431):

```python
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"\nLoading best checkpoint from:\n{best_ckpt_path}\n")

    if best_ckpt_path:
        best_model = PLModule.load_from_checkpoint(
            best_ckpt_path,
            config=config,
            class_names=train_ds.classes,
            recording_environment_class_names=environment_class_names,
            pretrained_checkpoint=pretrained_checkpoint,
            frame_hz=frame_hz,
        )
    else:
        best_model = model
    trainer.test(best_model, dataloaders=test_dl)
```
**Why:** When running short smoke tests (`--n_epochs < --check_val_every_n_epoch`), validation does not run and `best_ckpt_path` is `""`. Without this guard, `PLModule.load_from_checkpoint("")` tries to open the working directory as a file and raises `PermissionError: [Errno 13]`.

*(Also copy the helper scripts in `commands/` if starting from a fresh upstream clone.)*

---

## 4. Download Dataset & Pretrained Backbone

### Step 7: Download & Extract the RealDESED Dataset (`data/`)
Download the three dataset archives from Zenodo (`https://zenodo.org/records/20056072`) using `curl` (supports automatic retries and resume `-C -`):
```powershell
New-Item -ItemType Directory -Force -Path data\downloads

curl.exe -L --fail --retry 3 -C - -o data\downloads\validation.zip "https://zenodo.org/records/20056072/files/validation.zip?download=1"
curl.exe -L --fail --retry 3 -C - -o data\downloads\test.zip       "https://zenodo.org/records/20056072/files/test.zip?download=1"
curl.exe -L --fail --retry 3 -C - -o data\downloads\train.zip      "https://zenodo.org/records/20056072/files/train.zip?download=1"
```
Extract all three splits cleanly into `data\validation`, `data\test`, and `data\train`:
```powershell
python commands\dataset_manager.py
```
Expected layout after extraction:
* `data\train\` — 3,704 audio files + `metadata.csv` + `annotations.csv` + `annotations_raw.csv`
* `data\validation\` — 999 audio files + `metadata.csv` + `annotations.csv`
* `data\test\` — 1,007 audio files + `metadata.csv` + `annotations.csv`

### Step 8: Download Pretrained ATST-F Backbone (`resources/`)
```powershell
New-Item -ItemType Directory -Force -Path resources
curl.exe -L --fail --retry 3 -C - -o resources\ATST-F_strong_1.pt "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/ATST-F_strong_1.pt"
```
*(Optional: if skipped, `train.py` downloads `ATST-F_strong_1.pt` (~329 MB) automatically on first run.)*

> **Note:** `resources\ATST-F_strong_1.pt` provides the AudioSet Strong–pretrained ATST-Frame Transformer backbone only; its AudioSet classification heads are stripped at load time and replaced with a new 15-class linear head (`self.event_head`) trained on RealDESED.

### Step 9: Compute Annotator Quality Scores
Required for the `Weighted Soft` baseline (`--train_annotation_aggregation "Weighted Soft"`):
```powershell
python compute_annotator_score.py --root data
```
Outputs `data\annotator_quality\annotator_scores_train.json` and `annotator_scores_train.csv`.

---

## 5. Verification, Training & Inference

### Step 10: Single-Clip GPU Sanity Check
Confirm end-to-end audio loading, mel-spectrogram extraction, ATST-F Transformer forward pass, and sliding-window inference on GPU:
```powershell
python commands\verify_forward_and_eval.py
```

### Step 11: Run Training
Choose parameters matching your new machine's GPU VRAM and OS:

* **For 4 GB VRAM GPUs / Windows** (exact setup used on GTX 1650 — keeps effective batch size $12 \times 4 = 48$):
  ```powershell
  python commands\run_reproduction.py --n_epochs 100 --batch_size 12 --accumulate_grad 4
  ```
* **For $\ge 11$ GB VRAM GPUs (e.g., RTX 2080 Ti / Linux paper setup)**:
  ```bash
  export WANDB_MODE=offline
  python train.py \
    --dataset_path data \
    --output_dir runs \
    --experiment_name atst_f_weighted_soft \
    --batch_size 48 \
    --accumulate_grad_batches 1 \
    --precision 16 \
    --num_workers 8 \
    --check_val_every_n_epoch 5 \
    --train_annotation_aggregation "Weighted Soft" \
    --annotator_alpha 16 \
    --include_reviewed_train_files
  ```
  *(On Windows, prefer `--num_workers 0` to avoid DataLoader process-spawning overhead.)*

### Step 12: Run Test Inference
Once training completes and saves a `.ckpt` file under `runs\`, pass it to `run_inference.py` to generate calibrated RealDESED detections:
```powershell
python commands\run_inference.py --checkpoint runs\experiment_dumps\checkpoints\<best_model>.ckpt --limit all
```
*(Running `python commands\run_inference.py --limit 5` without `--checkpoint` verifies the inference pipeline using the `ATST-F_strong_1.pt` backbone with an untrained linear `event_head`.)*

---

## 6. Linux / macOS Quick Equivalent

If reproducing on Linux (with `conda` or `venv`):
* Use `python3.10 -m venv .venv && source .venv/bin/activate` (or `bash setup.sh` if using Conda, **then pin** `pip install "transformers==4.57.6"` if running `torch==2.5.1`).
* Run `export WANDB_MODE=offline` before launching `train.py` if not logging in to Weights & Biases.

---

## 7. Known Pitfalls & Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ImportError: get_cosine_schedule_with_warmup requires the PyTorch library` | `transformers >= 5.0` disables PyTorch integration when `torch < 2.6` | Install `transformers==4.57.6` (`<5.0`) |
| `torch.cuda.is_available()` is `False` on Windows | Standard `pip install torch` pulls CPU-only wheels on Windows | Install using `--index-url https://download.pytorch.org/whl/cu121` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `setuptools >= 81` removed `pkg_resources` | Pin `setuptools==65.5.0` (`<81`) |
| `train.py: error: argument --precision: invalid int value: '16-mixed'` | `train.py` defines `--precision` as `type=int` | Pass `--precision 16` (not `'16-mixed'`) |
| `PermissionError: [Errno 13]` after short training (`n_epochs < 5`) | `best_model_path` is `""` when validation never ran | Apply the `if best_ckpt_path:` patch from Section 3 |
| CUDA Out of Memory (OOM) during training | Default `--batch_size 48` needs >8 GB VRAM | Use `--batch_size 12 --accumulate_grad_batches 4 --precision 16` |
| Slow DataLoader / hangs on Windows | Default `--num_workers 16` spawns 16 Python processes on Windows | Pass `--num_workers 0` |
| Download stalls on Zenodo via Python `urllib` | Zenodo rate-limits / drops single-stream Python downloads | Use `curl -L --fail --retry 3 -C -` |

