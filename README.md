# Manako Vision AI Benchmark

Rigorous object detection benchmark comparing **SN44** (Bittensor subnet-trained) vs **SAM3** (pseudo ground-truth ceiling) vs **Roboflow** (best open-source) on **mAP@50**.

Demonstrates that web3 decentralized training via SN44 produces competitive vision models.

## Setup

```bash
pip install -e .

# For GPU inference:
pip install -e ".[cuda]"

# For local Roboflow inference:
pip install -e ".[roboflow]"
```

## Quick Start

### 1. Prepare your data

Place evaluation frames and annotations:

```
data/
  frames/           # .jpg/.png images
  annotations/
    annotations.json  # COCO format
```

Or use YOLO-format labels (one `.txt` per image in the annotations directory).

### 2. Run the benchmark

```bash
# SN44 only (auto-downloads weights from HuggingFace)
manako-bench run --frames data/frames --annotations data/annotations/annotations.json

# All three models
manako-bench run \
  --frames data/frames \
  --annotations data/annotations/annotations.json \
  --roboflow-key YOUR_KEY \
  --sam3-endpoint https://your-sam3-endpoint.com

# Specific models only
manako-bench run \
  --frames data/frames \
  --annotations data/annotations/annotations.json \
  --models sn44,roboflow \
  --roboflow-key YOUR_KEY

# With checkpoint tag for temporal tracking
manako-bench run \
  --frames data/frames \
  --annotations data/annotations/annotations.json \
  --checkpoint-tag epoch_50
```

### 3. View results

```bash
# Regenerate HTML report
manako-bench report

# View improvement history
manako-bench history

# Validate dataset
manako-bench validate --frames data/frames --annotations data/annotations/annotations.json
```

## Models

| Model | Source | Role |
|-------|--------|------|
| **SN44** | [alfred8995/kane001](https://huggingface.co/alfred8995/kane001) | Subnet-trained YOLO (ONNX, 960x960) |
| **SAM3** | API endpoint | Pseudo ground-truth ceiling |
| **Roboflow** | [vehicles-q0x2v](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v) | Best open-source baseline |

**Detection classes:** bus, car, truck, motorcycle

## Architecture

```
manako_benchmark/
  cli.py              # Click CLI entry point
  config.py           # YAML/env config loader
  data/
    dataset.py         # COCO & YOLO dataset loader
  models/
    base.py            # ModelAdapter interface
    sn44.py            # ONNX local inference (HuggingFace)
    sam3.py            # API-based inference
    roboflow.py        # Roboflow API or local SDK
  evaluation/
    metrics.py         # mAP@50 via pycocotools
    runner.py          # Benchmark orchestrator
    temporal.py        # Checkpoint-over-time tracker
  reporting/
    dashboard.py       # Plotly + Jinja2 HTML report
```

## Output

- `results/results_YYYY-MM-DD.json` — raw benchmark data
- `results/history.json` — temporal tracking across runs
- `reports/report_YYYY-MM-DD.html` — interactive HTML dashboard with:
  - Overall mAP@50 comparison bar chart
  - Per-class AP@50 grouped bars
  - Improvement over time line chart (if multiple checkpoints)
  - Ceiling gap analysis (SN44 vs SAM3)
  - Full results table with inference speed

## Configuration

Copy `config.example.yaml` to `config.yaml`, or use environment variables with `MANAKO_` prefix:

```bash
export MANAKO_ROBOFLOW_API_KEY=your_key
export MANAKO_SAM3_ENDPOINT=https://...
```
