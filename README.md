# pix2pix-seg2photo

Pix2Pix baseline for segmentation → photo synthesis using a U-Net generator and a 70×70 PatchGAN discriminator. Trains on side-by-side images where the **right half** is the segmentation mask and the **left half** is the target photo.

This repository mirrors the production layout used in the SPADE variant:
- Clean `src/` package (dataset, models, training script)
- Reproducible runs via Docker and docker-compose
- Notebooks live under `notebooks/` (outputs stripped)
- Data kept outside the repo; artifacts go to `outputs/`

---

## Features

- U-Net generator (8 down / 8 up) and 70×70 PatchGAN discriminator  
- Training direction: **seg → photo** (right half → left half)  
- Losses: BCEWithLogits (adv) + L1 reconstruction (`λ=100` by default)  
- Optional **feature matching** on D’s intermediate features  
- Deterministic seeding, clean folders for `checkpoints/`, `samples/`, `pred/`  
- Optional FID computation if `torch-fidelity` is available (best model is tracked)

---

## Data layout (kept outside the repo)

Each `.jpg` contains two halves on a single canvas:

- **left**: target **photo**  
- **right**: input **segmentation** (color-coded labels)

Images are cropped to `(256, 256)` halves internally.

```text
/path/to/data/
  train/
    train/
      1.jpg
      2.jpg
      ...
  test/
    test/
      1.jpg
      2.jpg
      ...
```

Keep data outside of version control. You will mount it into the container (or point local training to the path).

---

## Requirements

- Python 3.10+
- PyTorch 2.x and torchvision (install per <https://pytorch.org/get-started/>)
- Other Python deps in `requirements.txt`

---

## Quickstart

### A) Local (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Install torch/torchvision as recommended by PyTorch for your OS/CUDA

python src/train_pix2pix.py \
  --train_dir /path/to/data/train/train \
  --val_dir   /path/to/data/test/test \
  --outdir    outputs \
  --batch_size 4 \
  --epochs 200 \
  --lr 2e-4 \
  --lambda_l1 100 \
  --save_every 10 \
  --eval_every 50 \
  --fm_weight 10
```

### B) Docker

Build from the repository root:

```bash
docker build -t pix2pix-seg2photo .
```

Run with GPUs and mounted volumes:

```bash
docker run --gpus all --rm \
  -v "$(pwd)/outputs:/workspace/outputs" \
  -v "/ABSOLUTE/PATH/TO/data:/data:ro" \
  pix2pix-seg2photo \
  python src/train_pix2pix.py \
    --train_dir /data/train/train \
    --val_dir   /data/test/test \
    --outdir    /workspace/outputs \
    --batch_size 4 --epochs 200 --lr 2e-4 --lambda_l1 100 \
    --save_every 10 --eval_every 50 --fm_weight 10
```

### C) docker-compose

```yaml
version: "3.9"
services:
  train:
    build: .
    image: pix2pix-seg2photo:latest
    command: >
      python src/train_pix2pix.py
      --train_dir /data/train/train
      --val_dir   /data/test/test
      --outdir    /workspace/outputs
      --batch_size 4
      --epochs 200
      --lr 2e-4
      --lambda_l1 100
      --save_every 10
      --eval_every 50
      --fm_weight 10
    volumes:
      - ./outputs:/workspace/outputs
      - /ABSOLUTE/PATH/TO/data:/data:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

Run:

```bash
docker compose up --build
```

### D) Colab (demo cell)

```python
!nvidia-smi
!git clone https://github.com/patilgayatri22/pix2pix-seg2photo.git
%cd /content/pix2pix-seg2photo
!pip install -r requirements.txt

# Data (Drive or downloaded elsewhere)
from google.colab import drive; drive.mount('/content/drive')
TRAIN_DIR = "/content/drive/MyDrive/seg2photo_data/train/train"
VAL_DIR   = "/content/drive/MyDrive/seg2photo_data/test/test"

!python src/train_pix2pix.py \
  --train_dir "$TRAIN_DIR" \
  --val_dir   "$VAL_DIR" \
  --outdir    outputs \
  --batch_size 4 --epochs 50 --lr 2e-4 --lambda_l1 100 \
  --save_every 10 --eval_every 50 --fm_weight 10
```

---

## Outputs

- Checkpoints: `outputs/checkpoints/epoch_*.pt` and `best_model.pt`
- Samples: `outputs/samples/epoch_*.jpg` (qualitative)
- Validation predictions: `outputs/pred/*.jpg`

Track only `best_model.pt` via Git LFS (or attach to a Release). Keep the repo lean.

---

## Results

Lower is better for FID.

| ID | FID | MiFID |
|---:|----:|------:|
| 1  | 50.153942627512414 | 0.13429156 |

CSV:

```csv
ID,FID,MiFID
1,50.153942627512414,0.13429156
```

---

## Repository layout

```text
.
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── notebooks/
│   └── PixtoPix.ipynb
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train_pix2pix.py
│   └── utils.py
└── outputs/                   # created at runtime
    ├── checkpoints/
    ├── pred/
    └── samples/
```

---

## Notes

- Training uses **seg → photo**: condition = right half (seg), target = left half (photo).
- Default hyperparameters follow the original paper/typical settings: `lr=2e-4`, `β=(0.5, 0.999)`, `λ_L1=100`.
- Feature matching is optional (off if `--fm_weight 0`).
- FID requires `torch-fidelity`. If available, the script computes it during eval and saves `best_model.pt`.

---

## License

MIT
