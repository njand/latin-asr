from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from latin_asr.metrics import (
    create_compute_metrics_fn,
    normalize_latin_text,
    to_percentage,
)

# ---------------------------------------------------------------------------
# Tests for to_percentage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_val, expected",
    [
        (0.0, 0.0),
        (0.15, 15.0),
        (1.0, 100.0),
        (1.25, 125.0),
    ],
)
def test_to_percentage(input_val, expected):
    """Test scaling float metric ratios to standard percentage values."""
    assert to_percentage(input_val) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests for normalize_latin_text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_text, expected_normalized",
    [
        ("Rōma", "roma"),
        ("ĀĒĪŌŪāēīōū", "aeiouaeiou"),
        ("Juventas", "iuuentas"),
        ("Veni, vidi, vici", "ueni, uidi, uici"),
        ("JŪLIUS", "iulius"),
        ("Rēx vīvit", "rex uiuit"),
    ],
)
def test_normalize_latin_text(raw_text, expected_normalized):
    """Test macron removal, case-folding, and j/v -> i/u substitution."""
    assert normalize_latin_text(raw_text) == expected_normalized


def test_normalize_latin_text_preserves_other_chars():
    """Test that numbers and standard non-diacritic characters remain unaltered."""
    assert normalize_latin_text("Anno 2026 AD!") == "anno 2026 ad!"


# ---------------------------------------------------------------------------
# Tests for create_compute_metrics_fn
# ---------------------------------------------------------------------------

@patch("evaluate.load")
def test_compute_metrics_pipeline(mock_eval_load):
    mock_wer = MagicMock()
    mock_cer = MagicMock()

    mock_eval_load.side_effect = lambda metric: {"wer": mock_wer, "cer": mock_cer}[metric]

    # Return values based on arguments rather than call order to avoid fragile tests
    mock_wer.compute.side_effect = lambda predictions, references: (
        0.20 if predictions == ["Iūlius", "Vēni"] else 0.10
    )
    mock_cer.compute.side_effect = lambda predictions, references: (
        0.10 if predictions == ["Iūlius", "Vēni"] else 0.05
    )

    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0
    mock_processor.batch_decode.side_effect = [
        ["Iūlius", "Vēni"],
        ["Iūlius", "Vēnī"],
    ]

    compute_metrics = create_compute_metrics_fn(mock_processor)

    mock_pred = MagicMock()
    mock_pred.predictions = np.array([
        [[0.1, 0.9, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.1, 0.1, 0.8, 0.0]],
        [[0.0, 0.0, 0.1, 0.9], [0.5, 0.5, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
    ])
    mock_pred.label_ids = np.array([[1, 0, 2], [3, 1, -100]])

    results = compute_metrics(mock_pred)

    # 1. Verify batch_decode calls including keyword arguments
    assert mock_processor.batch_decode.call_count == 2
    
    # Check predictions decode (no group_tokens kwarg)
    np.testing.assert_array_equal(
        mock_processor.batch_decode.call_args_list[0].args[0],
        np.array([[1, 0, 2], [3, 0, 0]])
    )
    
    # Check labels decode (must pass group_tokens=False)
    np.testing.assert_array_equal(
        mock_processor.batch_decode.call_args_list[1].args[0],
        np.array([[1, 0, 2], [3, 1, 0]])
    )
    assert mock_processor.batch_decode.call_args_list[1].kwargs == {"group_tokens": False}

    # 2. Explicitly verify WER AND CER calls (Strict & Normalized)
    expected_strict = (["Iūlius", "Vēni"], ["Iūlius", "Vēnī"])
    expected_norm = (["iulius", "ueni"], ["iulius", "ueni"])

    for metric in (mock_wer, mock_cer):
        metric.compute.assert_any_call(predictions=expected_strict[0], references=expected_strict[1])
        metric.compute.assert_any_call(predictions=expected_norm[0], references=expected_norm[1])

    # 3. Verify final output structure
    assert results == {
        "strict_wer": 0.20,
        "strict_cer": 0.10,
        "norm_wer": 0.10,
        "norm_cer": 0.05,
    }