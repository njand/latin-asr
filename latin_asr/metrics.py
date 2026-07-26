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
    """Factory function returning the evaluation callback for Strict and Normalized WER/CER metrics."""
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace ignored index padding (-100) with pad token ID for decoding
        label_ids = np.where(
            pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids
        )

        # Decode exact model outputs and target labels (Strict)
        pred_str: list[str] = processor.batch_decode(pred_ids)
        label_str: list[str] = processor.batch_decode(label_ids, group_tokens=False)

        # Derive normalized strings (Strip macrons & map j/v -> i/u)
        norm_pred_str = [normalize_latin_text(s) for s in pred_str]
        norm_label_str = [normalize_latin_text(s) for s in label_str]

        # Compute Strict metrics
        strict_wer = wer_metric.compute(predictions=pred_str, references=label_str)
        strict_cer = cer_metric.compute(predictions=pred_str, references=label_str)

        # Compute Normalized metrics
        norm_wer = wer_metric.compute(predictions=norm_pred_str, references=norm_label_str)
        norm_cer = cer_metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return {
            "strict_wer": strict_wer,
            "strict_cer": strict_cer,
            "norm_wer": norm_wer,
            "norm_cer": norm_cer
        }

    return compute_metrics