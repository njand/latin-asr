from unittest.mock import patch

from latin_asr.templates import (
    build_readme_table,
    generate_final_readme,
    generate_fold_readme,
)

# ---------------------------------------------------------------------------
# Tests for build_readme_table
# ---------------------------------------------------------------------------

def test_build_readme_table_empty_eval():
    """Test that build_readme_table returns an empty string when no eval metrics exist."""
    log_history = [
        {"step": 10, "loss": 0.5, "epoch": 0.5},
        {"step": 20, "loss": 0.4, "epoch": 1.0},
    ]
    assert build_readme_table(log_history) == ""


@patch("latin_asr.templates.to_percentage", side_effect=lambda x: round(x * 100, 2))
def test_build_readme_table_formatting_and_best_step(mock_to_pct):
    """Test correctly formatted table headers, rows, prior train loss matching, and best step bolding."""
    log_history = [
        {"step": 10, "loss": 0.85432, "epoch": 0.5},
        {
            "step": 20,
            "epoch": 1.0,
            "eval_loss": 0.4500,
            "eval_strict_wer": 0.20,
            "eval_strict_cer": 0.10,
            "eval_norm_wer": 0.15,
            "eval_norm_cer": 0.08,
        },
        {"step": 25, "loss": 0.32101, "epoch": 1.25},
        {
            "step": 30,
            "epoch": 1.5,
            "eval_loss": 0.2500,  # Best step (lowest val_loss)
            "eval_strict_wer": 0.12,
            "eval_strict_cer": 0.05,
            "eval_norm_wer": 0.09,
            "eval_norm_cer": 0.03,
        },
    ]

    table_md = build_readme_table(log_history)

    # Check header
    assert "| Epoch | Step | Train Loss | Val Loss | Strict WER | Strict CER | Norm WER | Norm CER |" in table_md

    # Row 1 (step 20): train loss corresponds to step 10 (0.8543)
    assert "| 1 | 20 | 0.8543 | 0.4500 | 20.00% | 10.00% | 15.00% | 8.00% |" in table_md

    # Row 2 (step 30 - Best Step): train loss corresponds to step 25 (0.3210) and is bolded
    assert "| **2** | **30** | **0.3210** | **0.2500** | **12.00%** | **5.00%** | **9.00%** | **3.00%** |" in table_md


@patch("latin_asr.templates.to_percentage", side_effect=lambda x: round(x * 100, 2))
def test_build_readme_table_fallback_metrics_and_no_prior_train_loss(mock_to_pct):
    """Test fallbacks for missing metrics (default 1.0 -> 100.0%) and missing prior train loss ("N/A")."""
    log_history = [
        {
            "step": 10,
            "epoch": 1.0,
            "eval_loss": 0.5000,
            # Strict/Norm WER/CER intentionally omitted
        }
    ]

    table_md = build_readme_table(log_history)

    # Should fall back to N/A for train loss and 100.00% for missing evaluation ratios
    assert "| **1** | **10** | **N/A** | **0.5000** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |" in table_md


# ---------------------------------------------------------------------------
# Tests for generate_fold_readme
# ---------------------------------------------------------------------------

def test_generate_fold_readme_placeholders():
    """Test fold model card string generation with expected metadata and table injection."""
    sample_table = "| 1 | 10 | 0.5000 | 0.4000 | 10.00% | 5.00% | 8.00% | 4.00% |"

    readme = generate_fold_readme(
        fold=2,
        base_model="facebook/wav2vec2-xls-r-300m",
        dataset_name="user/latin_ds",
        table_md=sample_table,
    )

    # Check YAML frontmatter and title headers
    assert "base_model: facebook/wav2vec2-xls-r-300m" in readme
    assert "# Wav2Vec2 XLS-R 300M - Latin ASR (Fold 2)" in readme
    assert "trained on **Fold 2** of [`user/latin_ds`]" in readme
    assert sample_table in readme


# ---------------------------------------------------------------------------
# Tests for generate_final_readme
# ---------------------------------------------------------------------------

def test_generate_final_readme_formatting():
    """Test final model card string generation with multi-fold CV table and Quickstart snippets."""
    fold_results = [
        {"fold": 0, "strict_wer": 12.5, "strict_cer": 5.2, "norm_wer": 9.1, "norm_cer": 3.0},
        {"fold": 1, "strict_wer": 11.5, "strict_cer": 4.8, "norm_wer": 8.5, "norm_cer": 2.8},
    ]

    readme = generate_final_readme(
        base_model="facebook/wav2vec2-xls-r-300m",
        dataset_name="user/latin_ds",
        hf_repo_id="user/wav2vec2-latin-300m",
        fold_results=fold_results,
        hours_str="2h 15m",
        carbon_str="0.05 kg CO2eq",
        avg_strict_wer=12.00,
        avg_strict_cer=5.00,
        avg_norm_wer=8.80,
        avg_norm_cer=2.90,
    )

    # Check cross-validation fold rows
    assert "| **Fold 0** | 12.50% | 5.20% | 9.10% | 3.00% |" in readme
    assert "| **Fold 1** | 11.50% | 4.80% | 8.50% | 2.80% |" in readme

    # Check summary metrics
    assert "| **Average (CV)** | **12.00%** | **5.00%** | **8.80%** | **2.90%** |" in readme

    # Check environmental impact section
    assert "Total Training Time:** 2h 15m" in readme
    assert "Estimated Carbon Emissions:** 0.05 kg CO2eq" in readme

    # Check Quickstart repo ID insertion
    assert 'pipeline("automatic-speech-recognition", model="user/wav2vec2-latin-300m")' in readme
    assert 'AutoProcessor.from_pretrained("user/wav2vec2-latin-300m")' in readme