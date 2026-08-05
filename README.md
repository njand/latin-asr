# `latin-asr`: Latin Automatic Speech Recognition

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Card-yellow)](https://huggingface.co/njand/wav2vec2-xls-r-latin)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Gradio-Live%20Demo-blue)](https://huggingface.co/spaces/njand/latin-asr-demo)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An end-to-end Machine Learning pipeline and Python framework for fine-tuning self-supervised speech representations ([`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m)) on a corpus of spoken Latin ([`njand/llpsi-speech-dataset`](https://huggingface.co/datasets/njand/llpsi-speech-dataset)). 

The system targets **Restored Classical Pronunciation** and features an automated serverless training pipeline built on **Modal**, **Hugging Face Hub**, **Weights & Biases**, and **CodeCarbon**.

---

## 🚀 Model Artifacts & Deployment

The fine-tuned model is published on Hugging Face in PyTorch and optimized ONNX formats:
👉 **[njand/wav2vec2-xls-r-latin](https://huggingface.co/njand/wav2vec2-xls-r-latin)** | 🎮 **[Live Gradio Demo](https://huggingface.co/spaces/njand/latin-asr-demo)**

### Model Formats & Memory Footprint

| Format | Precision | Storage Footprint | Target Environment |
| :--- | :--- | :--- | :--- |
| **PyTorch** | FP32 | 1.26 GB | Training & GPU Inference |
| **ONNX** | FP32 | 1.26 GB | Cross-Platform Runtimes |
| **ONNX Quantized** | **INT8** | **355 MB** | **Low-Latency CPU & Edge Production** |

---

## ⚡ Quickstart

### Standard PyTorch Inference

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="njand/wav2vec2-xls-r-latin")
result = transcriber("path/to/latin.wav")
print(result["text"])

```

### High-Efficiency CPU Inference (Optimum ONNX INT8)

For production deployments on CPU hardware without CUDA dependencies:

```bash
pip install "optimum[onnxruntime]"

```

```python
from optimum.onnxruntime import ORTModelForCTC
from transformers import AutoProcessor

# Load 355 MB INT8 quantized ONNX weights
model = ORTModelForCTC.from_pretrained(
    "njand/wav2vec2-xls-r-latin", 
    file_name="model_quantized.onnx"
)
processor = AutoProcessor.from_pretrained("njand/wav2vec2-xls-r-latin")

# Run inference
inputs = processor("path/to/latin.wav", return_tensors="pt")
logits = model(**inputs).logits

```

---

## 📊 Cross-Validation Performance

Evaluated across a 5-fold cross-validation scheme on `njand/llpsi-speech-dataset`. Metrics track both **Strict** text recognition (preserving macron diacritics and *j/v* glides) and **Normalized** text recognition (macrons stripped, *j* to *i*, *v* to *u*).

| Fold Metric | Strict WER | Strict CER | Norm WER | Norm CER |
| --- | --- | --- | --- | --- |
| **Fold 0** | 27.09% | 13.06% | 25.26% | 12.48% |
| **Fold 1** | 26.33% | 12.92% | 24.61% | 12.33% |
| **Fold 2** | 26.27% | 12.74% | 24.59% | 12.18% |
| **Fold 3** | 27.21% | 12.91% | 25.26% | 12.27% |
| **Fold 4** | 28.16% | 13.12% | 26.59% | 12.58% |
| **Average (5-Fold CV)** | **27.01%** | **12.95%** | **25.26%** | **12.37%** |

---

## 💡 Engineering Highlights

* **Serverless GPU Orchestration:** Remote multi-GPU fine-tuning executed on NVIDIA L4 instances via **Modal**, featuring persistent volume caching (`latin-audio-cache`) for rapid dataset staging.
* **Dual-Metric Linguistic Evaluator:** Custom evaluators track both phonological surface forms (macrons & *j/v* distinction) and normalized underlying forms during validation steps.
* **Hugging Face Hub Lifecycle Automation:** Programmatically provisions fold-specific Git branches (e.g., `fold-0`, `fold-1`) on Hugging Face Hub, pushing checkpoint weights, trainer states, evaluation tables, and generated model cards.
* **Production Quantization Pipeline:** Exports trained PyTorch models to ONNX INT8, reducing memory footprint by **72%** (355 MB) while accelerating CPU inference by ~3x.
* **Telemetry & Carbon Tracking:** Measures operational carbon footprint using `codecarbon` (total run emissions: ~0.31 kg CO2) and syncs offline W&B runs post-training.

---

## 🛠️ Repository Layout

```text
latin-asr/
├── .github/
│   └── workflows/           # GitHub Actions CI suite (linting & tests)
├── apps/
│   ├── train.py             # Modal orchestrator: 5-fold CV & full dataset fit
│   └── wipe_modal_cache.py  # Maintenance utility for persistent Modal storage
├── latin_asr/               # Core package module
├── scripts/                 # Core package module
│   └── export_onnx.py       # Convert and quantize trained model to ONNX format
├── tests/                   # Comprehensive pytest suite
├── pyproject.toml           # Package metadata & tool configs (Ruff, Pytest)
└── README.md

```

---

## ⚙️ Development & Local Setup

```bash
# Clone repository
git clone [https://github.com/njand/latin-asr.git](https://github.com/njand/latin-asr.git)
cd latin-asr

# Install editable package with dev tools
pip install -e .[dev]

# Run tests and linter
pytest
ruff check .

```

### Remote Cloud Execution (Modal)

Ensure Modal secrets `huggingface-secret` (Write token required) and `wandb-secret` are configured before launching jobs:

```bash
# Launch 5-fold cross-validation & final model fit
modal run apps/train.py

# Reset remote persistent storage (optional)
modal run apps/wipe_modal_cache.py

```

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](https://github.com/njand/latin-asr/blob/main/LICENSE) for details.
