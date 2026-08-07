import unicodedata
from collections.abc import Callable
from typing import Any

import evaluate
import numpy as np


def to_percentage(raw_val: float) -> float:
    """Converts a raw metric ratio (e.g. 0.15 or 1.25) directly to percentage (15.0 or 125.0)."""
    return float(raw_val) * 100.0


def normalize_latin_text(text: str) -> str:
    """Strips macrons/diacritics and converts consonantal j/v to vocalic i/u."""
    text = text.lower()
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return (
        unicodedata.normalize("NFC", stripped)
        .replace("j", "i")
        .replace("v", "u")
    )


def create_compute_metrics_fn(processor: Any) -> Callable[[Any], dict[str, float]]:
    """Factory function returning an optimized evaluation callback for Strict and Normalized WER/CER metrics."""
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Decode model outputs (CTC grouping active)
        pred_str: list[str] = processor.batch_decode(
            pred_ids, 
            skip_special_tokens=True
        )

        # Vectorized / native list conversion for fast iteration
        label_ids = (
            pred.label_ids.tolist()
            if isinstance(pred.label_ids, np.ndarray)
            else pred.label_ids
        )
        pad_id = processor.tokenizer.pad_token_id

        # Clean target labels: filter out loss-ignore mask (-100) and pad tokens
        label_ids_cleaned = [
            [token for token in label if token != -100 and (pad_id is None or token != pad_id)]
            for label in label_ids
        ]

        # Decode target labels preserving double letters (group_tokens=False)
        label_str: list[str] = processor.batch_decode(
            label_ids_cleaned, 
            group_tokens=False, 
            skip_special_tokens=True
        )

        # 5. Derive normalized strings
        norm_pred_str = [normalize_latin_text(s) for s in pred_str]
        norm_label_str = [normalize_latin_text(s) for s in label_str]

        # 6. Safeguard: Prevent division-by-zero crashes on empty references
        label_str_safe = [s if s.strip() else " " for s in label_str]
        norm_label_str_safe = [s if s.strip() else " " for s in norm_label_str]

        # 7. Compute Strict & Normalized metrics
        strict_wer = wer_metric.compute(predictions=pred_str, references=label_str_safe)
        strict_cer = cer_metric.compute(predictions=pred_str, references=label_str_safe)
        norm_wer = wer_metric.compute(predictions=norm_pred_str, references=norm_label_str_safe)
        norm_cer = cer_metric.compute(predictions=norm_pred_str, references=norm_label_str_safe)

        return {
            "strict_wer": float(strict_wer),
            "strict_cer": float(strict_cer),
            "norm_wer": float(norm_wer),
            "norm_cer": float(norm_cer),
        }

    return compute_metrics