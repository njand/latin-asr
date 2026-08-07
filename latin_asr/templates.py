from typing import Any

from latin_asr.metrics import to_percentage


def build_readme_table(log_history: list[dict[str, Any]]) -> str:
    """Constructs a formatted Markdown metrics table from trainer log history."""
    train_losses = {entry["step"]: entry["loss"] for entry in log_history if "loss" in entry}
    eval_metrics = {entry["step"]: entry for entry in log_history if "eval_loss" in entry}

    if not eval_metrics:
        return ""

    # Identify the best step based on lowest validation loss
    best_step = min(eval_metrics.keys(), key=lambda s: eval_metrics[s].get("eval_loss", float("inf")))

    table_rows = []
    for step in sorted(eval_metrics.keys()):
        e = eval_metrics[step]
        epoch = round(e.get("epoch", 0.0))
        val_loss = e.get("eval_loss", 0.0)

        # Extract strict and normalized metrics (falling back to raw wer/cer if missing)
        strict_wer = to_percentage(e.get("eval_strict_wer", 1.0))
        strict_cer = to_percentage(e.get("eval_strict_cer", 1.0))
        norm_wer = to_percentage(e.get("eval_norm_wer", 1.0))
        norm_cer = to_percentage(e.get("eval_norm_cer", 1.0))

        # Get the most recent training loss logged at or before this evaluation step
        prior_steps = [s for s in train_losses if s <= step]
        tr_str = f"{train_losses[max(prior_steps)]:.4f}" if prior_steps else "N/A"

        cols = [
            f"{epoch}",
            f"{step}",
            tr_str,
            f"{val_loss:.4f}",
            f"{strict_wer:.2f}%",
            f"{strict_cer:.2f}%",
            f"{norm_wer:.2f}%",
            f"{norm_cer:.2f}%",
        ]

        # Highlight the best step in bold
        if step == best_step:
            cols = [f"**{c}**" for c in cols]

        table_rows.append(f"| {' | '.join(cols)} |")

    header = (
        "| Epoch | Step | Train Loss | Val Loss | Strict WER | Strict CER | Norm WER | Norm CER |\n"
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    )
    return header + "\n" + "\n".join(table_rows)


def generate_fold_readme(fold: int, base_model: str, dataset_name: str, table_md: str) -> str:
    """Generates a concise Model Card README string for an individual cross-validation fold."""
    return f"""---
language:
- la
license: mit
base_model: {base_model}
tags:
- automatic-speech-recognition
- wav2vec2
- latin
- classical-latin
- macrons
- audio
datasets:
- {dataset_name}
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# Wav2Vec2 XLS-R 300M - Latin ASR (Fold {fold})

Fine-tuned version of [`{base_model}`](https://huggingface.co/{base_model}) for Latin Automatic Speech Recognition, trained on **Fold {fold}** of [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name}).

- **Pronunciation Standard:** Restored Classical
- **Orthography:** Fully Macronized (ā, ē, ī, ō, ū, ȳ) with consonantal **j/v**

---

## 📊 Training History & Metrics

{table_md}

> **Note:** **Strict** metrics evaluate exact macron placement and **j/v** orthography. **Norm** (Normalized) metrics evaluate core word recognition after stripping macrons (ā → a) and standardizing glides (j → i, v → u).
"""


def generate_final_readme(
    base_model: str,
    dataset_name: str,
    hf_repo_id: str,
    fold_results: list[dict[str, Any]],
    hours_str: str,
    carbon_str: str,
    avg_strict_wer: float,
    avg_strict_cer: float,
    avg_norm_wer: float,
    avg_norm_cer: float
) -> str:
    """Generates the consolidated Model Card README string for the final training run."""
    fold_table_rows = "\n".join(
        [
            f"| **Fold {f['fold']}** | {f['strict_wer']:.2f}% | {f['strict_cer']:.2f}% | {f['norm_wer']:.2f}% | {f['norm_cer']:.2f}% |"
            for f in fold_results
        ]
    )
    fold_count = len(fold_results)

    return f"""---
language:
- la
license: mit
base_model: {base_model}
tags:
- automatic-speech-recognition
- wav2vec2
- latin
- classical-latin
- macrons
- audio
- onnx
- optimum
- int8
- gradio
datasets:
- {dataset_name}
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
model-index:
- name: Wav2Vec2 XLS-R 300M - Latin ASR
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      type: {dataset_name}
      name: LLPSI Speech Dataset
    metrics:
    - name: 5-Fold CV Strict WER
      type: wer
      value: {avg_strict_wer:.2f}
    - name: 5-Fold CV Strict CER
      type: cer
      value: {avg_strict_cer:.2f}
    - name: 5-Fold CV Normalized WER
      type: wer
      value: {avg_norm_wer:.2f}
    - name: 5-Fold CV Normalized CER
      type: cer
      value: {avg_norm_cer:.2f}
---

# Wav2Vec2 XLS-R 300M - Latin ASR (Restored Classical Pronunciation)

Fine-tuned version of [`{base_model}`](https://huggingface.co/{base_model}) for Latin Automatic Speech Recognition (ASR), trained on reading passages from `{dataset_name}`.

- **Live Demo:** [Gradio Interface](https://huggingface.co/spaces/njand/latin-asr-demo)
- **Source Code:** [GitHub Repository](https://github.com/njand/latin-asr)
- **Base Model:** [`{base_model}`](https://huggingface.co/{base_model})
- **Dataset:** [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name}) (currently private)
- **Pronunciation Standard:** Restored Classical Pronunciation
- **Orthography:** Macrons (ā, ē, ī, ō, ū, ȳ); consonantal j/v

---

## 🛠️ Model Variants & Optimization

To facilitate production deployment on CPU-based infrastructure, this repository provides the model in three formats:

| Format | Precision | File Size | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **PyTorch** | FP32 | --- | Training, fine-tuning, and research |
| **ONNX** | FP32 | --- | Cross-platform inference |
| **ONNX Quantized** | INT8 | --- | **Production, Edge, & CPU Inference** |

> **Why Quantized?** The INT8 version offers ~3x faster inference speed on standard CPUs compared to the original FP32 PyTorch weights while retaining near-identical transcription accuracy.

---

## 🚀 Quickstart

### Option 1: Standard Inference (PyTorch)

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="{hf_repo_id}")
result = transcriber("sample.wav")
print(result["text"])

```

### Option 2: Optimized Inference (Optimum ONNX Runtime)

For CPU inference, I recommend using the INT8 quantized model. Install dependencies: `pip install "optimum[onnxruntime]"`

```python
from optimum.onnxruntime import ORTModelForCTC
from transformers import AutoProcessor

# Load the quantized model
model = ORTModelForCTC.from_pretrained(
    "{hf_repo_id}", 
    file_name="model_quantized.onnx"
)
processor = AutoProcessor.from_pretrained("{hf_repo_id}")

# Inference pipeline
inputs = processor("sample.wav", return_tensors="pt")
outputs = model(**inputs)
# ... decode outputs

```

---

## 🔤 Orthography & Text Normalization

This model transcribes audio using the following orthographic conventions:

* **Vowel Quantity:** Macronizes long vowels (ā, ē, ī, ō, ū, ȳ).
* **Consonantal vs. Vocalic Glides:** Distinguishes consonantal **j** and **v** from vocalic **i** and **u** (e.g., *ējiciō* rather than *eicio*, *vīvus* rather than *uiuus*).
* **Casing & Punctuation:** Transcribes lowercase text with no punctuation.

---

## 📊 Cross-Validation Performance

Evaluated across {fold_count}-fold cross-validation on `{dataset_name}`:

| Fold | Strict WER | Strict CER | Norm WER | Norm CER |
| --- | --- | --- | --- | --- |
| {fold_table_rows} |  |  |  |  |
| **Average (CV)** | **{avg_strict_wer:.2f}%** | **{avg_strict_cer:.2f}%** | **{avg_norm_wer:.2f}%** | **{avg_norm_cer:.2f}%** |

> **Note:** Normalized WER/CER measure core word recognition (macrons stripped, j/v → i/u), while Strict WER/CER enforce exact macron placement and **j/v** orthography.

---

## ⚡ Environmental Impact & Compute

* **Hardware:** NVIDIA L4 GPU (24GB VRAM) via Modal
* **Total Training Time:** {hours_str}
* **Estimated Carbon Emissions:** {carbon_str}

---

## ⚠️ Limitations & Out-of-Scope Use

* **Pronunciation Variances:** Performance will drop on audio using **Ecclesiastical / Italianate** pronunciation (e.g., pronouncing *c* before *e/i* as `/tʃ/` rather than `/k/`).
* **Audio Conditions:** Optimized for clear, single-speaker reading. Background noise or overlapping speakers will reduce accuracy significantly.
"""
