import os
import subprocess
from typing import Any

import modal
import numpy as np

from latin_asr.collator import DataCollatorCTCWithPadding

# Import core package logic
from latin_asr.config import TrainingConfig
from latin_asr.dataset import get_processor, load_and_prepare_dataset
from latin_asr.metrics import create_compute_metrics_fn
from latin_asr.templates import build_readme_table, generate_final_readme, generate_fold_readme
from latin_asr.trainer_utils import (
    ModalProgressLogger,
    execute_training_step,
    parse_emissions_summary,
)

# ---------------------------------------------------------------------------
# Modal Container & Mount Setup
# ---------------------------------------------------------------------------
app = modal.App("wav2vec2-latin-cv-pipeline")

training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "datasets[audio]",
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "accelerate",
        "librosa",
        "jiwer",
        "evaluate",
        "soundfile",
        "huggingface_hub",
        "pandas",
        "numpy",
        "wandb",
        "codecarbon"
    )
    .add_local_python_source("latin_asr")
)

cache_volume = modal.Volume.from_name("latin-audio-cache", create_if_missing=True)


# ANSI Colors for High-Visibility Structural Headers
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_BG_MAGENTA = "\033[30;45m"


# ---------------------------------------------------------------------------
# Step Helpers
# ---------------------------------------------------------------------------
def run_fold(
    fold: int,
    config: TrainingConfig,
    processed_ds: Any,
    processor: Any,
    data_collator: Any,
    compute_metrics: Any,
    api: Any,
    hf_token: str
) -> tuple[float, dict]:
    """Runs cross-validation for a single fold using the general step runner."""
    from transformers import EarlyStoppingCallback

    print(f"\n{CLR_BOLD}{CLR_BG_MAGENTA} =================== FOLD {fold} / {config.num_folds - 1} =================== {CLR_RESET}", flush=True)
    branch_name = f"fold-{fold}"
    train_ds = processed_ds.filter(lambda x: x["fold"] != fold).remove_columns(["fold"])
    eval_ds = processed_ds.filter(lambda x: x["fold"] == fold).remove_columns(["fold"])

    def fold_readme_builder(trainer, eval_res):
        table_md = build_readme_table(trainer.state.log_history) if trainer else ""
        return generate_fold_readme(fold, config.base_model, config.dataset_name, table_md)

    res = execute_training_step(
        config=config,
        branch_name=branch_name,
        run_name=f"{config.exp_prefix}-fold-{fold}",
        target_epochs=config.target_fold_epochs,
        output_dir=f"/mnt/cache/output_fold_{fold}",
        train_ds=train_ds,
        eval_ds=eval_ds,
        processor=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ModalProgressLogger()],
        api=api,
        hf_token=hf_token,
        readme_content_fn=fold_readme_builder,
        commit_message=f"Completed training fold {fold}"
    )

    cache_volume.commit()
    return res["best_epoch"], {"fold": fold, "strict_wer": res["strict_wer"], "strict_cer": res["strict_cer"], "norm_wer": res["norm_wer"], "norm_cer": res["norm_cer"]}


def run_final_fit(
    config: TrainingConfig,
    target_epochs: int,
    processed_ds: Any,
    processor: Any,
    data_collator: Any,
    fold_results: list[dict],
    api: Any,
    hf_token: str,
    tracker: Any
) -> None:
    """Runs final training pass over all data using average optimal epoch count."""
    print("\n=================== FINAL MODEL RUN (ALL DATA) ===================", flush=True)
    full_ds = processed_ds.remove_columns(["fold"])

    avg_strict_wer = sum(f["strict_wer"] for f in fold_results) / len(fold_results) if fold_results else 0.0
    avg_strict_cer = sum(f["strict_cer"] for f in fold_results) / len(fold_results) if fold_results else 0.0
    avg_norm_wer = sum(f["norm_wer"] for f in fold_results) / len(fold_results) if fold_results else 0.0
    avg_norm_cer = sum(f["norm_cer"] for f in fold_results) / len(fold_results) if fold_results else 0.0

    def final_readme_builder(trainer, eval_res):
        # Force CodeCarbon to flush latest metrics to emissions.csv
        if tracker is not None:
            tracker.flush()
            
        hours_str, carbon_str = parse_emissions_summary()
        
        return generate_final_readme(
            base_model=config.base_model,
            dataset_name=config.dataset_name,
            hf_repo_id=config.hf_repo_id,
            fold_results=fold_results,
            hours_str=hours_str,
            carbon_str=carbon_str,
            avg_strict_wer=avg_strict_wer,
            avg_strict_cer=avg_strict_cer,
            avg_norm_wer=avg_norm_wer,
            avg_norm_cer=avg_norm_cer
        )

    execute_training_step(
        config=config,
        branch_name="main",
        run_name=f"{config.exp_prefix}-final-fit",
        target_epochs=target_epochs,
        output_dir="/mnt/cache/output_final",
        train_ds=full_ds,
        eval_ds=None,
        processor=processor,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[ModalProgressLogger()],
        api=api,
        hf_token=hf_token,
        readme_content_fn=final_readme_builder,
        commit_message="Completed final model fit on all data"
    )

    cache_volume.commit()


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
@app.function(
    image=training_image,
    gpu="L4",
    timeout=86400,
    volumes={"/mnt/cache": cache_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ]
)
def run_pipeline(config: TrainingConfig | None = None):
    from codecarbon import EmissionsTracker
    from huggingface_hub import HfApi

    if "WANDB_API_KEY" not in os.environ:
        raise ValueError("WANDB_API_KEY secret was not found in environment!")

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/mnt/cache/wandb"  # Ensure logs persist on volume
    os.makedirs("/mnt/cache/wandb", exist_ok=True)

    config = config or TrainingConfig()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Ensure Modal secret is attached.")

    api = HfApi(token=hf_token)
    tracker = EmissionsTracker(output_dir=".", output_file="emissions.csv", log_level="warning")
    tracker.start()

    try:
        # 1. Pipeline Prep
        processor = get_processor(config.hf_repo_id, config.base_model, hf_token)
        compute_metrics = create_compute_metrics_fn(processor)
        data_collator = DataCollatorCTCWithPadding(processor=processor)
        processed_ds = load_and_prepare_dataset(processor, config.dataset_name, hf_token)

        # 2. Cross-Validation Phase
        best_epochs, fold_results = [], []
        for fold in range(config.num_folds):
            opt_epoch, f_res = run_fold(
                fold, config, processed_ds, processor, data_collator, compute_metrics, api, hf_token
            )
            best_epochs.append(opt_epoch)
            fold_results.append(f_res)

        # 3. Final Full-Dataset Fit Phase
        avg_optimal_epoch = float(np.mean(best_epochs)) if best_epochs else config.target_fold_epochs
        final_target_epochs = max(1, int(np.round(avg_optimal_epoch)))

        run_final_fit(
            config, final_target_epochs, processed_ds, processor, data_collator, fold_results, api, hf_token, tracker
        )

    finally:
        try:
            tracker.stop()
        except Exception as e:  # noqa: BLE001
            print(f"Failed to stop carbon tracker: {e}")

    print("\n☁️ Syncing offline W&B runs to cloud...", flush=True)
    try:
        subprocess.run(
            ["wandb", "sync", "--sync-all"],
            cwd="/mnt/cache/wandb",
            check=True
        )
        print("✅ W&B sync complete.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ W&B sync failed: {e}", flush=True)

        
@app.local_entrypoint()
def main():
    run_pipeline.spawn()