import gc
import os
import time
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_callback import PrinterCallback

from latin_asr.config import TrainingConfig
from latin_asr.hf_utils import ensure_branch_exists, get_branch_progress, get_latest_checkpoint
from latin_asr.metrics import to_percentage
from latin_asr.models import build_model

# Standard ANSI Color Codes (Supported by Modal & Unix Terminals)
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_DIM = "\033[90m"
CLR_GREEN = "\033[32m"
CLR_CYAN = "\033[36m"
CLR_YELLOW = "\033[33m"


class ModalProgressLogger(TrainerCallback):
    """Custom progress logger for Modal console output with space-track bar, ANSI colors, and evaluation summaries."""

    def __init__(self, bar_width: int = 20):
        self.start_time = None
        self.bar_width = bar_width
        # Sub-block elements in 1/8ths increments
        self.partials = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Helper to format seconds into a clean human-readable duration."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m:02d}m {s:02d}s"

    def _render_bar(self, current_step: int, total_steps: int) -> str:
        """Renders green sub-block bar transitioning cleanly into empty space background."""
        total_eighths = int((current_step / total_steps) * self.bar_width * 8)
        full_blocks = total_eighths // 8
        remainder = total_eighths % 8
        empty_blocks = self.bar_width - full_blocks - (1 if remainder else 0)

        filled = "█" * full_blocks + self.partials[remainder]
        empty = " " * max(0, empty_blocks)

        return f"{CLR_GREEN}{filled}{CLR_RESET}{empty}"

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{CLR_BOLD}{CLR_CYAN}🚀 Training started...{CLR_RESET}\n", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and state.max_steps > 0:
            current_step = state.global_step
            total_steps = state.max_steps
            pct = (current_step / total_steps) * 100

            # Render custom space-track bar
            bar = self._render_bar(current_step, total_steps)

            # Timing calculations
            now = time.time()
            elapsed = now - (self.start_time or now)
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            remaining_steps = total_steps - current_step
            eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            elapsed_str = self._format_time(elapsed)
            eta_str = self._format_time(eta_sec)

            # Extract metrics & format raw strings for alignment
            epoch_val = state.epoch if state.epoch is not None else 0.0
            loss_val = logs.get("loss")
            loss_raw = f"{loss_val:6.4f}" if loss_val is not None else f"{'N/A':>6}"

            step_digits = len(str(total_steps))
            step_raw = f"{current_step:>{step_digits}}/{total_steps}"

            # Apply ANSI formatting around pre-padded strings
            pct_fmt = f"{CLR_BOLD}{pct:5.1f}%{CLR_RESET}"
            epoch_fmt = f"Epoch {CLR_CYAN}{epoch_val:5.2f}{CLR_RESET}"
            step_fmt = f"Step {CLR_CYAN}{step_raw}{CLR_RESET}"
            loss_fmt = f"Loss: {CLR_YELLOW}{loss_raw}{CLR_RESET}"
            time_fmt = f"{CLR_DIM}Elapsed: {elapsed_str:>8} | ETA: {eta_str:>8}{CLR_RESET}"

            print(
                f"{CLR_DIM}[{CLR_RESET}{bar}{CLR_DIM}]{CLR_RESET} {pct_fmt} | {epoch_fmt} | {step_fmt} | {loss_fmt} | {time_fmt}",
                flush=True,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Fires whenever evaluation completes at the end of an epoch."""
        if state.is_world_process_zero and metrics:
            epoch = metrics.get("epoch", 0.0)
            eval_loss = metrics.get("eval_loss", 0.0)

            strict_wer = to_percentage(metrics.get("eval_strict_wer", 1.0))
            strict_cer = to_percentage(metrics.get("eval_strict_cer", 1.0))
            norm_wer = to_percentage(metrics.get("eval_norm_wer", 1.0))
            norm_cer = to_percentage(metrics.get("eval_norm_cer", 1.0))

            print(
                f"\n{CLR_BOLD}{CLR_CYAN}📊 [Eval Epoch {epoch:.1f}]{CLR_RESET} Val Loss: {CLR_YELLOW}{eval_loss:.4f}{CLR_RESET}\n"
                f"   ├─ Strict: WER {CLR_BOLD}{strict_wer:5.2f}%{CLR_RESET} | CER {CLR_BOLD}{strict_cer:5.2f}%{CLR_RESET}\n"
                f"   └─ Norm:   WER {CLR_BOLD}{norm_wer:5.2f}%{CLR_RESET} | CER {CLR_BOLD}{norm_cer:5.2f}%{CLR_RESET}\n",
                flush=True,
            )


def build_training_args(
    output_dir: str,
    config: TrainingConfig,
    num_epochs: int,
    branch_name: str,
    hf_token: str,
    run_name: str,
    is_eval_enabled: bool,
) -> TrainingArguments:
    """Constructs TrainingArguments for Hugging Face Trainer."""
    eval_strategy = "epoch" if is_eval_enabled else "no"
    save_strategy = "epoch"

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        num_train_epochs=num_epochs,
        logging_steps=config.logging_steps,
        disable_tqdm=True,
        dataloader_num_workers=4,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        remove_unused_columns=False,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        save_total_limit=2,
        push_to_hub=False,
        hub_token=hf_token,
        hub_revision=branch_name,
        hub_strategy="end",
        hub_model_id=config.hf_repo_id,
        run_name=run_name,
        load_best_model_at_end=is_eval_enabled,
        metric_for_best_model="eval_loss" if is_eval_enabled else None,
        greater_is_better=False,
        report_to="wandb"
    )
    

def publish_model_and_readme(
    trainer: Trainer,
    api: HfApi,
    readme_content: str,
    output_dir: str,
    hf_repo_id: str,
    branch_name: str,
    commit_msg: str,
) -> None:
    """Saves model artifacts, updates README.md, and uploads files to target HF Hub branch."""
    # Ensure latest model and processor state are saved to disk
    trainer.save_model(output_dir)
    trainer.save_state(output_dir)
    trainer.processing_class.save_pretrained(output_dir)

    # Write the pre-generated README (YAML frontmatter + body)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Push complete directory to specified branch with commit message
    api.upload_folder(
        folder_path=output_dir,
        repo_id=hf_repo_id,
        revision=branch_name,
        commit_message=commit_msg,
        ignore_patterns=["checkpoint-*"],  # <-- Exclude local checkpoints from HF upload
    )


def parse_emissions_summary(emissions_csv_path: str = "emissions.csv") -> tuple[str, str]:
    """Parses CodeCarbon output file and returns formatted hours and carbon emissions strings."""
    if not os.path.exists(emissions_csv_path):
        return "N/A", "N/A"

    try:
        import pandas as pd
        df = pd.read_csv(emissions_csv_path)
        total_duration_sec = df["duration"].sum() if "duration" in df else 0.0
        total_emissions_kg = df["emissions"].sum() if "emissions" in df else 0.0

        hours = total_duration_sec / 3600.0
        hours_str = f"{hours:.2f} hours"
        carbon_str = f"{total_emissions_kg:.4f} kg CO2eq"
        return hours_str, carbon_str
    except Exception:  # noqa: BLE001
        return "N/A", "N/A"


def execute_training_step(
    config: TrainingConfig,
    branch_name: str,
    run_name: str,
    target_epochs: int,
    output_dir: str,
    train_ds: Any,
    eval_ds: Any | None,
    processor: Any,
    data_collator: Any,
    compute_metrics: Any | None,
    callbacks: list[Any] | None,
    api: HfApi,
    hf_token: str,
    readme_content_fn: Any,  # Function that receives (trainer, eval_results) -> str
    commit_message: str
) -> dict[str, Any]:
    """Generic runner that handles W&B setup, branch tracking, model training, and Hub uploads."""
    import wandb

    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        project="wav2vec2-latin",
        id=run_name.replace(".", "_"),
        name=run_name,
        reinit="finish_previous"
    )

    ensure_branch_exists(api, config.hf_repo_id, branch_name)
    completed_epochs, best_epoch, saved_strict_wer, saved_strict_cer, saved_norm_wer, saved_norm_cer = get_branch_progress(
        api, config.hf_repo_id, branch_name, hf_token
    )

    # Short-circuit if this branch/fold was completed in a previous run
    if completed_epochs >= target_epochs:
        print(f"-> Branch '{branch_name}' previously completed. Preserving historical metrics.", flush=True)
        return {
            "completed_previously": True,
            "best_epoch": best_epoch if best_epoch is not None else target_epochs,
            "strict_wer": saved_strict_wer,
            "strict_cer": saved_strict_cer,
            "norm_wer": saved_norm_wer,
            "norm_cer": saved_norm_cer
        }

    resume_ckpt = get_latest_checkpoint(output_dir)
    model_checkpoint = (
        config.hf_repo_id if (completed_epochs > 0 and not resume_ckpt) else config.base_model
    )
    revision_target = branch_name if (completed_epochs > 0 and not resume_ckpt) else None

    model = build_model(
        config,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        checkpoint=model_checkpoint,
        revision=revision_target,
    )

    is_eval_enabled = eval_ds is not None
    training_args = build_training_args(
        output_dir=output_dir,
        config=config,
        num_epochs=target_epochs,
        branch_name=branch_name,
        hf_token=hf_token,
        run_name=run_name,
        is_eval_enabled=is_eval_enabled,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if is_eval_enabled else None,
        callbacks=callbacks
    )

    trainer.remove_callback(PrinterCallback)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    eval_results = {}
    if is_eval_enabled:
        best_log = min(
            [log for log in trainer.state.log_history if "eval_loss" in log],
            key=lambda x: x["eval_loss"]
        )
        eval_results["strict_wer"] = to_percentage(best_log.get("eval_strict_wer", 1.0))
        eval_results["strict_cer"] = to_percentage(best_log.get("eval_strict_cer", 1.0))
        eval_results["norm_wer"] = to_percentage(best_log.get("eval_norm_wer", 1.0))
        eval_results["norm_cer"] = to_percentage(best_log.get("eval_norm_cer", 1.0))

    # Generate model card & publish to Hugging Face
    readme_text = readme_content_fn(trainer, eval_results)
    publish_model_and_readme(
        trainer=trainer,
        api=api,
        readme_content=readme_text,
        output_dir=training_args.output_dir,
        hf_repo_id=config.hf_repo_id,
        branch_name=branch_name,
        commit_msg=commit_message
    )

    _, updated_best_epoch, _, _, _, _ = get_branch_progress(api, config.hf_repo_id, branch_name, hf_token)

    if wandb.run is not None:
        wandb.finish()

    # Garbage collection
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "completed_previously": False,
        "best_epoch": updated_best_epoch if updated_best_epoch is not None else target_epochs,
        "strict_wer": eval_results.get("strict_wer", 0.0),
        "strict_cer": eval_results.get("strict_cer", 0.0),
        "norm_wer": eval_results.get("norm_wer", 0.0),
        "norm_cer": eval_results.get("norm_cer", 0.0)
    }