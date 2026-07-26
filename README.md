# Latin Automatic Speech Recognition (`latin-asr`)

An end-to-end pipeline for fine-tuning Wav2Vec2 architectures (e.g., `facebook/wav2vec2-xls-r-300m`) on Classical Latin speech corpora. The framework leverages **Modal** for cloud GPU execution, **Hugging Face Hub** for model tracking, **Weights & Biases** for experiment logging, and **CodeCarbon** for environmental impact measurement.

---

## 🛠️ Repository Layout

```text
latin-asr/
├── .github/
│   └── workflows/          # GitHub Actions CI configuration
├── apps/
│   ├── train.py            # Modal entrypoint: N-fold CV & final full-dataset fit
│   └── wipe_modal_cache.py # Utility script to clean persistent Modal storage
├── latin_asr/              # Core package (dataset, collator, models, metrics, trainer)
├── tests/                  # Unit test suite
├── pyproject.toml          # PEP 621 package metadata & configuration
└── README.md

```

---

## ✨ Key Features

* **Serverless Cloud Execution:** Fine-tunes on remote NVIDIA L4 GPUs via Modal using persistent volume caching (`latin-audio-cache`).

* **Automated N-Fold Cross-Validation:** Trains across 5 folds, determines average optimal early-stopping epochs, and runs a final fit on the full dataset.

* **Dual Metric Evaluation:** Evaluates both **Strict** text recognition (preserving diacritics/macrons and $j/v$ glides) and **Normalized** text recognition (macrons stripped, $j \to i, v \to u$).

* **Hugging Face Hub Branching:** Automatically manages per-fold git branches (e.g., `fold-0`, `fold-1`), publishing model weights, evaluation tables, and custom model cards.

* **Telemetry & Environmental Tracking:** Logs carbon footprint via `codecarbon` and syncs offline Weights & Biases runs post-training.

---

## ⚙️ Prerequisites & Modal Secrets

To run remote training on Modal, set up the following secrets in your Modal workspace:

1. **`huggingface-secret`**: Must contain an `HF_TOKEN` environment variable.

> **Note:** The Hugging Face token **must have Write permissions** because the pipeline dynamically creates git branches (`fold-0`, `fold-1`, etc.) and uploads model weights, trainer states, and model cards.
> 
> 

2. **`wandb-secret`**: Must contain a `WANDB_API_KEY` environment variable for logging.

---

## ⚡ Quickstart

### 1. Local Environment Setup

Clone the repository and install the package with dev dependencies:

```bash
git clone https://github.com/njand/latin-asr.git
cd latin-asr
pip install -e .[dev]

```

### 2. Run Tests & Linter

Run the full unit test suite and linting rules:

```bash
# Run pytest suite
pytest

# Run linter
ruff check .

```

### 3. Remote Cloud Training via Modal

Launch the 5-fold cross-validation and final model training job on Modal GPUs:

```bash
modal run apps/train.py

```

### 4. Clear Remote Storage (Optional)

To wipe the persistent Modal volume cache (`/mnt/cache`):

```bash
modal run apps/wipe_modal_cache.py

```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.