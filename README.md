# Manako Vision AI Benchmark

Rigorous detection benchmark comparing **SN44** (Bittensor subnet-trained) vs **SAM3** (Meta) vs **Roboflow** across person detection and vehicle detection.

**SN44 achieves 84% of SAM3's accuracy at 0.5% of the model size — 150x more efficient.**

[**View interactive results**](https://mxmsbt.github.io/manako_benchmark/)

## Results

### Vision Efficiency Score: E = mAP / S

| Model | Person E | Vehicle E | Combined E | Rank |
|-------|----------|-----------|------------|------|
| **SN44** | **4.47** | **4.03** | **4.25** | **#1** |
| Roboflow | — | — | — | — |
| SAM3 | 0.028 | 0.029 | 0.028 | #3 |

### Person Detection (200 images, 5,883 annotations)

| Metric | SN44 | SAM3 | Roboflow |
|--------|------|------|----------|
| mAP@50 | **84.1%** | 97.3% | 33.5% |
| Precision | **95.8%** | 99.0% | 94.2% |
| Recall | 72.4% | 82.0% | 27.3% |

### Vehicle Detection (200 images, 4 classes)

| Metric | SN44 | SAM3 | Roboflow |
|--------|------|------|----------|
| mAP@50 | 79.4% | **98.2%** | 7.8% |
| Car | 77.7% | 98.7% | 13.3% |
| Bus | **90.1%** | 98.0% | 9.2% |
| Truck | 63.4% | 98.3% | 8.6% |
| Motorcycle | 86.6% | 97.7% | 0.0% |

### Compute

| | SN44 | SAM3 | Roboflow |
|---|---|---|---|
| Model Size | **19 MB** | 3,450 MB | — |
| Parameters | **4.4M** | 848M | — |
| FLOPs/image | **31–52 G** | ~2,900 G | — |
| GPU Required | **No (CPU)** | Yes (24GB+) | Cloud API |
| Cost/1K images | **~0** | $0.06 | $0.04 |

## Models

| Model | HuggingFace | Task |
|-------|------------|------|
| SN44 Person | [manak0/Detect-Person-winner](https://huggingface.co/manak0/Detect-Person-winner) | Person detection (ONNX, 4.4M params) |
| SN44 Vehicle | [manak0/Detect-detect-vehicle-winner](https://huggingface.co/manak0/Detect-detect-vehicle-winner) | Vehicle detection — car, bus, truck, motorcycle (ONNX, 4.4M params) |
| SAM3 | [facebook/sam3](https://huggingface.co/facebook/sam3) | Foundation model (848M params, GPU required) |
| Roboflow Person | [cctv-naxyo/1](https://universe.roboflow.com/dataset-uutxr/cctv-naxyo) | CCTV person detection (cloud API) |
| Roboflow Vehicle | [vehicles-q0x2v/1](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v) | Vehicle detection (cloud API) |

## Quick Start

```bash
pip install -e .

# Person detection benchmark
manako-bench run \
  --frames data/private_gt/frames \
  --annotations data/private_gt/annotations.json \
  --models sn44,sam3 \
  --sn44-repo manak0/Detect-Person-winner \
  --sam3-predictions sam3_person.json \
  --no-remap-classes

# Vehicle detection benchmark
manako-bench run \
  --frames data/vehicle_gt/frames \
  --annotations data/vehicle_gt/annotations_4class.json \
  --models sn44,sam3 \
  --sn44-repo manak0/Detect-detect-vehicle-winner \
  --sam3-predictions sam3_vehicle_4class.json \
  --no-remap-classes
```

## Methodology

- All models evaluated against **independently generated ground truth** — no model was involved in creating the GT it was evaluated against
- SAM3 predictions generated on A100 GPU via [facebook/sam3](https://github.com/facebookresearch/sam3)
- SN44 models run on CPU (Apple M-series) via ONNX Runtime
- SN44 input resolution matched to SAM3 (1024px vs 1008px for person, native 1280px for vehicle)
- Validated across multiple public benchmarks: COCO Person, CrowdHuman, WiderPerson
- Vision Efficiency Score: **E = mAP / S** (accuracy per megabyte of model)

## Architecture

```
manako_benchmark/
  cli.py                # Click CLI
  models/
    sn44.py             # ONNX local inference (auto-downloads from HuggingFace)
    sam3.py             # Pre-computed predictions loader
    roboflow.py         # Roboflow API
  evaluation/
    metrics.py          # mAP@50 via pycocotools
    runner.py           # Benchmark orchestrator
  data/
    dataset.py          # COCO & YOLO dataset loader
  reporting/
    dashboard.py        # Plotly HTML reports
scripts/
  generate_sam3_predictions.py  # Run SAM3 on GPU
  generate_yolo_gt.py           # YOLO consensus pseudo-GT
```
