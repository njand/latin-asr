from unittest.mock import MagicMock, patch

import pytest

from latin_asr.models import build_model


@pytest.fixture
def mock_config():
    """Fixture providing a mock TrainingConfig object with dummy hyperparameters."""
    config = MagicMock()
    config.attention_dropout = 0.1
    config.hidden_dropout = 0.1
    config.feat_proj_dropout = 0.0
    config.final_dropout = 0.1
    config.layerdrop = 0.05
    config.mask_time_prob = 0.05
    config.mask_time_length = 10
    config.mask_feature_prob = 0.0
    config.mask_feature_length = 10
    config.freeze_feature_encoder = False
    return config


# ---------------------------------------------------------------------------
# Tests for build_model
# ---------------------------------------------------------------------------

@patch("transformers.Wav2Vec2ForCTC.from_pretrained")
def test_build_model_hyperparameter_forwarding(mock_from_pretrained, mock_config):
    """Test that build_model passes all configuration parameters to Wav2Vec2ForCTC.from_pretrained."""
    mock_model_instance = MagicMock()
    mock_from_pretrained.return_value = mock_model_instance

    model = build_model(
        config=mock_config,
        pad_token_id=0,
        vocab_size=32,
        checkpoint="facebook/wav2vec2-base",
        revision="main",
    )

    # Verify return value
    assert model == mock_model_instance

    # Verify call parameters
    mock_from_pretrained.assert_called_once_with(
        "facebook/wav2vec2-base",
        revision="main",
        ignore_mismatched_sizes=True,
        ctc_loss_reduction="mean",
        pad_token_id=0,
        vocab_size=32,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.05,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
    )

    # Feature encoder should NOT be frozen when config.freeze_feature_encoder is False
    mock_model_instance.freeze_feature_encoder.assert_not_called()


@patch("transformers.Wav2Vec2ForCTC.from_pretrained")
def test_build_model_freeze_feature_encoder(mock_from_pretrained, mock_config):
    """Test that freeze_feature_encoder() is called on the model when enabled in config."""
    mock_model_instance = MagicMock()
    mock_from_pretrained.return_value = mock_model_instance
    mock_config.freeze_feature_encoder = True

    build_model(
        config=mock_config,
        pad_token_id=0,
        vocab_size=32,
        checkpoint="facebook/wav2vec2-base",
    )

    # Verify freeze_feature_encoder was triggered
    mock_model_instance.freeze_feature_encoder.assert_called_once()


@patch("transformers.Wav2Vec2ForCTC.from_pretrained")
def test_build_model_default_optional_revision(mock_from_pretrained, mock_config):
    """Test that revision defaults to None when omitted."""
    mock_from_pretrained.return_value = MagicMock()

    build_model(
        config=mock_config,
        pad_token_id=0,
        vocab_size=32,
        checkpoint="facebook/wav2vec2-base",
    )

    # Check that revision=None was passed in kwargs
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["revision"] is None