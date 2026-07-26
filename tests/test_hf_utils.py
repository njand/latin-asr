import json
from unittest.mock import MagicMock, patch

import pytest

from latin_asr.hf_utils import (
    ensure_branch_exists,
    get_branch_progress,
    get_latest_checkpoint,
)

# ---------------------------------------------------------------------------
# Tests for ensure_branch_exists
# ---------------------------------------------------------------------------

def test_ensure_branch_exists_success():
    """Test successful branch creation or verification."""
    mock_api = MagicMock()

    ensure_branch_exists(mock_api, "user/repo", "experimental")

    mock_api.create_branch.assert_called_once_with(
        repo_id="user/repo", branch="experimental", exist_ok=True
    )


def test_ensure_branch_exists_handles_exception(capsys):
    """Test that exceptions during branch creation are caught and printed gracefully."""
    mock_api = MagicMock()
    mock_api.create_branch.side_effect = Exception("API connection error")

    # Should not raise an exception
    ensure_branch_exists(mock_api, "user/repo", "experimental")

    captured = capsys.readouterr()
    assert "Warning: Could not verify/create branch 'experimental': API connection error" in captured.out


# ---------------------------------------------------------------------------
# Tests for get_latest_checkpoint
# ---------------------------------------------------------------------------

def test_get_latest_checkpoint_no_checkpoints(tmp_path):
    """Test return value when no checkpoint directories exist."""
    assert get_latest_checkpoint(str(tmp_path)) is None


def test_get_latest_checkpoint_selects_highest_step(tmp_path):
    """Test that the checkpoint with the highest numerical step is selected."""
    # Create sample checkpoint folders
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-250").mkdir()

    latest = get_latest_checkpoint(str(tmp_path))

    assert latest == str(tmp_path / "checkpoint-500")


def test_get_latest_checkpoint_filters_invalid_names(tmp_path):
    """Test that directories matching pattern but lacking integer steps are ignored."""
    (tmp_path / "checkpoint-invalid").mkdir()
    (tmp_path / "checkpoint-150").mkdir()

    latest = get_latest_checkpoint(str(tmp_path))

    assert latest == str(tmp_path / "checkpoint-150")


def test_get_latest_checkpoint_all_invalid(tmp_path):
    """Test return value when checkpoint folders exist but none have valid integer suffixes."""
    (tmp_path / "checkpoint-abc").mkdir()

    assert get_latest_checkpoint(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tests for get_branch_progress
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_trainer_state_data():
    """Fixture supplying sample trainer_state.json dictionary contents."""
    return {
        "epoch": 5.0,
        "log_history": [
            {
                "epoch": 2.0,
                "eval_loss": 0.45,
                "eval_strict_wer": 0.20,
                "eval_strict_cer": 0.10,
                "eval_norm_wer": 0.15,
                "eval_norm_cer": 0.08,
            },
            {
                "epoch": 4.0,
                "eval_loss": 0.30,  # Best loss
                "eval_strict_wer": 0.12,
                "eval_strict_cer": 0.05,
                "eval_norm_wer": 0.09,
                "eval_norm_cer": 0.03,
            },
        ],
    }


@patch("huggingface_hub.hf_hub_download")
@patch("latin_asr.metrics.to_percentage", side_effect=lambda x: round(x * 100, 2))
def test_get_branch_progress_root_state_file(
    mock_to_pct, mock_download, tmp_path, mock_trainer_state_data
):
    """Test parsing when trainer_state.json is located at the repository root."""
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["README.md", "trainer_state.json"]

    # Write state to temporary file to simulate hf_hub_download local return
    state_file_path = tmp_path / "trainer_state.json"
    state_file_path.write_text(json.dumps(mock_trainer_state_data))
    mock_download.return_value = str(state_file_path)

    epochs, best_epoch, s_wer, s_cer, n_wer, n_cer = get_branch_progress(
        mock_api, "user/repo", "main", "fake_token"
    )

    assert epochs == 5.0
    assert best_epoch == 4.0
    assert s_wer == 12.0
    assert s_cer == 5.0
    assert n_wer == 9.0
    assert n_cer == 3.0


@patch("huggingface_hub.hf_hub_download")
@patch("latin_asr.metrics.to_percentage", side_effect=lambda x: x * 100)
def test_get_branch_progress_nested_checkpoint_state(
    mock_to_pct, mock_download, tmp_path, mock_trainer_state_data
):
    """Test finding the latest checkpoint trainer_state.json when root file is missing."""
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = [
        "checkpoint-100/trainer_state.json",
        "checkpoint-300/trainer_state.json",
        "checkpoint-200/trainer_state.json",
        "checkpoint-invalid/trainer_state.json",
    ]

    state_file_path = tmp_path / "trainer_state.json"
    state_file_path.write_text(json.dumps(mock_trainer_state_data))
    mock_download.return_value = str(state_file_path)

    get_branch_progress(mock_api, "user/repo", "main", "fake_token")

    # Verify that checkpoint-300 (highest step) was chosen for download
    mock_download.assert_called_once_with(
        repo_id="user/repo",
        filename="checkpoint-300/trainer_state.json",
        revision="main",
        token="fake_token",
    )


def test_get_branch_progress_no_state_files():
    """Test fallback when no state files exist in the repository."""
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["README.md", "model.safetensors"]

    res = get_branch_progress(mock_api, "user/repo", "main", "fake_token")
    assert res == (0.0, None, 0.0, 0.0, 0.0, 0.0)


@patch("huggingface_hub.hf_hub_download")
def test_get_branch_progress_no_eval_logs(mock_download, tmp_path):
    """Test handling state history with training logs but no evaluation logs."""
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["trainer_state.json"]

    state_data = {
        "epoch": 2.5,
        "log_history": [{"epoch": 1.0, "loss": 0.8}, {"epoch": 2.0, "loss": 0.5}],
    }
    state_file_path = tmp_path / "trainer_state.json"
    state_file_path.write_text(json.dumps(state_data))
    mock_download.return_value = str(state_file_path)

    res = get_branch_progress(mock_api, "user/repo", "main", "fake_token")
    assert res == (2.5, 2.5, 0.0, 0.0, 0.0, 0.0)


def test_get_branch_progress_exception_handling():
    """Test that API/network exceptions are caught and return default zero values."""
    mock_api = MagicMock()
    mock_api.list_repo_files.side_effect = Exception("404 Branch Not Found")

    res = get_branch_progress(mock_api, "user/repo", "missing_branch", "fake_token")
    assert res == (0.0, None, 0.0, 0.0, 0.0, 0.0)