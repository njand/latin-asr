from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latin_asr.trainer_utils import (
    ModalProgressLogger,
    build_training_args,
    execute_training_step,
    parse_emissions_summary,
    publish_model_and_readme,
)


@pytest.fixture
def mock_training_config():
    config = MagicMock()
    config.hf_repo_id = "user/repo"
    config.base_model = "facebook/wav2vec2-base"
    config.bf16 = False
    config.fp16 = False
    config.per_device_train_batch_size = 8
    config.per_device_eval_batch_size = 8
    config.gradient_accumulation_steps = 2
    config.learning_rate = 1e-4
    config.weight_decay = 0.01
    config.warmup_steps = 500

    return config


# ---------------------------------------------------------------------------
# Tests for ModalProgressLogger
# ---------------------------------------------------------------------------

def test_logger_format_time():
    """Test duration formatting into human-readable strings."""
    logger = ModalProgressLogger()
    assert logger._format_time(45) == "00m 45s"
    assert logger._format_time(125) == "02m 05s"
    assert logger._format_time(3665) == "1h 01m 05s"


def test_logger_render_bar():
    """Test space-track sub-block bar generation."""
    logger = ModalProgressLogger(bar_width=10)

    # 0% completion -> empty string
    bar_0 = logger._render_bar(0, 100)
    assert "█" not in bar_0

    # 100% completion -> full blocks
    bar_100 = logger._render_bar(100, 100)
    assert "█" * 10 in bar_100


def test_logger_on_log_output(capsys):
    """Test logging output rendering on step intervals."""
    logger = ModalProgressLogger(bar_width=10)
    logger.start_time = 100.0

    mock_state = MagicMock()
    mock_state.is_world_process_zero = True
    mock_state.max_steps = 100
    mock_state.global_step = 50
    mock_state.epoch = 1.0

    logs = {"loss": 0.4521}

    with patch("time.time", return_value=200.0):
        logger.on_log(args=MagicMock(), state=mock_state, control=MagicMock(), logs=logs)

    captured = capsys.readouterr()
    assert "50.0%" in captured.out
    assert "Epoch" in captured.out
    assert "1.00" in captured.out
    assert "Step" in captured.out
    assert "50/100" in captured.out
    assert "0.4521" in captured.out


def test_logger_on_evaluate_output(capsys):
    """Test evaluation summary metrics printed at epoch completion."""
    logger = ModalProgressLogger()
    mock_state = MagicMock()
    mock_state.is_world_process_zero = True

    metrics = {
        "epoch": 2.0,
        "eval_loss": 0.3500,
        "eval_strict_wer": 0.15,
        "eval_strict_cer": 0.05,
        "eval_norm_wer": 0.10,
        "eval_norm_cer": 0.03,
    }

    with patch("latin_asr.trainer_utils.to_percentage", side_effect=lambda x: x * 100):
        logger.on_evaluate(args=MagicMock(), state=mock_state, control=MagicMock(), metrics=metrics)

    captured = capsys.readouterr()
    assert "[Eval Epoch 2.0]" in captured.out
    assert "Val Loss:" in captured.out
    assert "0.3500" in captured.out
    assert "15.00%" in captured.out
    assert "5.00%" in captured.out


# ---------------------------------------------------------------------------
# Tests for build_training_args
# ---------------------------------------------------------------------------

@patch("latin_asr.trainer_utils.TrainingArguments")
def test_build_training_args_eval_enabled(mock_training_args, mock_training_config):
    """Test TrainingArguments configuration when validation evaluation is enabled."""
    build_training_args(
        output_dir="/tmp/test_out",
        config=mock_training_config,
        num_epochs=3,
        branch_name="fold-0",
        hf_token="fake_token",
        run_name="test_run",
        is_eval_enabled=True,
    )

    # Verify that TrainingArguments was instantiated with expected kwargs
    mock_training_args.assert_called_once()
    _, kwargs = mock_training_args.call_args
    assert kwargs["train_sampling_strategy"] == "group_by_length"
    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["load_best_model_at_end"] is True


def test_build_training_args_eval_disabled(mock_training_config):
    """Test TrainingArguments configuration when evaluation is disabled."""
    args = build_training_args(
        output_dir="/tmp/test_out",
        config=mock_training_config,
        num_epochs=3,
        branch_name="main",
        hf_token="fake_token",
        run_name="test_run",
        is_eval_enabled=False,
    )
    
    assert args.eval_strategy.value == "no"


# ---------------------------------------------------------------------------
# Tests for publish_model_and_readme
# ---------------------------------------------------------------------------

def test_publish_model_and_readme(tmp_path):
    """Test saving model state, generating README.md, and invoking Hugging Face upload."""
    mock_trainer = MagicMock()
    mock_api = MagicMock()

    out_dir = str(tmp_path)
    readme_content = "# Model README Card"

    publish_model_and_readme(
        trainer=mock_trainer,
        api=mock_api,
        readme_content=readme_content,
        output_dir=out_dir,
        hf_repo_id="user/repo",
        branch_name="main",
        commit_msg="Update model",
    )

    # Verify model and processor save triggers
    mock_trainer.save_model.assert_called_once_with(out_dir)
    mock_trainer.processing_class.save_pretrained.assert_called_once_with(out_dir)

    # Verify README file was created on disk
    written_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert written_readme == readme_content

    # Verify upload_folder parameters
    mock_api.upload_folder.assert_called_once_with(
        folder_path=out_dir,
        repo_id="user/repo",
        revision="main",
        commit_message="Update model",
        ignore_patterns=["checkpoint-*"],
    )


# ---------------------------------------------------------------------------
# Tests for parse_emissions_summary
# ---------------------------------------------------------------------------

def test_parse_emissions_summary_missing_file():
    """Test fallback values when CodeCarbon CSV file does not exist."""
    hours, carbon = parse_emissions_summary("non_existent_emissions.csv")
    assert hours == "N/A"
    assert carbon == "N/A"


def test_parse_emissions_summary_valid_csv(tmp_path):
    """Test parsing accumulated duration and CO2 emissions from CodeCarbon output."""
    csv_file = tmp_path / "emissions.csv"
    df = pd.DataFrame(
        {
            "duration": [3600.0, 1800.0],  # 5400 sec = 1.50 hours
            "emissions": [0.025, 0.015],    # 0.0400 kg
        }
    )
    df.to_csv(csv_file, index=False)

    hours, carbon = parse_emissions_summary(str(csv_file))

    assert hours == "1.50 hours"
    assert carbon == "0.0400 kg CO2eq"


# ---------------------------------------------------------------------------
# Tests for execute_training_step
# ---------------------------------------------------------------------------

@patch("wandb.init")
@patch("wandb.finish")
@patch("latin_asr.trainer_utils.get_branch_progress")
@patch("latin_asr.trainer_utils.ensure_branch_exists")
def test_execute_training_step_short_circuit(
    mock_ensure_branch,
    mock_get_progress,
    mock_wandb_finish,
    mock_wandb_init,
    mock_training_config,  # Injected fixture goes at the end of the parameters
):
    """Test that training is skipped if the target branch was already completed in a prior run."""
    mock_api = MagicMock()

    # Simulate previously completed run (5 completed out of 5 target epochs)
    mock_get_progress.return_value = (5.0, 4.0, 10.0, 5.0, 8.0, 4.0)

    result = execute_training_step(
        config=mock_training_config,  # Use fixture directly here
        branch_name="fold-0",
        run_name="run_fold_0",
        target_epochs=5,
        output_dir="/tmp/out",
        train_ds=MagicMock(),
        eval_ds=MagicMock(),
        processor=MagicMock(),
        data_collator=MagicMock(),
        compute_metrics=MagicMock(),
        callbacks=[],
        api=mock_api,
        hf_token="fake_token",
        readme_content_fn=lambda t, e: "readme",
        commit_message="Finished",
    )

    assert result["completed_previously"] is True
    assert result["best_epoch"] == 4.0
    assert result["strict_wer"] == 10.0


@patch("wandb.init")
@patch("wandb.finish")
@patch("latin_asr.trainer_utils.get_latest_checkpoint", return_value=None)
@patch("latin_asr.trainer_utils.get_branch_progress", return_value=(0.0, None, 0.0, 0.0, 0.0, 0.0))
@patch("latin_asr.trainer_utils.ensure_branch_exists")
@patch("latin_asr.trainer_utils.build_model")
@patch("latin_asr.trainer_utils.Trainer")
@patch("latin_asr.trainer_utils.publish_model_and_readme")
def test_execute_training_step_full_run(
    mock_publish,
    mock_trainer_cls,
    mock_build_model,
    mock_ensure_branch,
    mock_get_progress,
    mock_get_ckpt,
    mock_wandb_finish,
    mock_wandb_init,
    mock_training_config
):
    """Test full training execution pipeline including evaluation and artifact publishing."""
    mock_api = MagicMock()

    mock_trainer_instance = MagicMock()
    mock_trainer_cls.return_value = mock_trainer_instance

    # Mock evaluation history inside trainer state
    mock_trainer_instance.state.log_history = [
        {"epoch": 1.0, "eval_loss": 0.5, "eval_strict_wer": 0.20, "eval_strict_cer": 0.10, "eval_norm_wer": 0.15, "eval_norm_cer": 0.08}
    ]

    mock_processor = MagicMock()
    mock_processor.tokenizer.__len__.return_value = 32

    result = execute_training_step(
        config=mock_training_config,
        branch_name="fold-1",
        run_name="run_fold_1",
        target_epochs=3,
        output_dir="/tmp/out",
        train_ds=MagicMock(),
        eval_ds=MagicMock(),
        processor=mock_processor,
        data_collator=MagicMock(),
        compute_metrics=MagicMock(),
        callbacks=[],
        api=mock_api,
        hf_token="fake_token",
        readme_content_fn=lambda t, e: "generated readme",
        commit_message="Finished fold 1",
    )

    # Verify model building, trainer execution, and publishing steps
    mock_build_model.assert_called_once()
    mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_publish.assert_called_once()

    # Verify result payload mapping
    assert result["completed_previously"] is False
    assert result["strict_wer"] == 20.0
    assert result["strict_cer"] == 10.0
    assert result["norm_wer"] == 15.0
    assert result["norm_cer"] == 8.0