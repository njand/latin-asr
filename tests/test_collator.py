from unittest.mock import MagicMock

import pytest
import torch

from latin_asr.collator import DataCollatorCTCWithPadding


@pytest.fixture
def mock_processor():
    """Fixture providing a mocked Hugging Face CTC processor."""
    processor = MagicMock()

    # Mock audio processing behavior: returns a dictionary with PyTorch tensors
    def mock_processor_call(audio_arrays, sampling_rate, padding, return_tensors):
        return {
            "input_values": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }

    processor.side_effect = mock_processor_call

    # Mock tokenizer padding behavior
    def mock_pad(label_features, padding, return_tensors):
        # Simulates padding input_ids [1, 2] and [3] to length 2 with pad_token_id = 0
        return {
            "input_ids": torch.tensor([[1, 2], [3, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }

    processor.tokenizer.pad.side_effect = mock_pad
    return processor


@pytest.fixture
def sample_features():
    """Fixture providing sample input dataset features."""
    return [
        {
            "audio": {"array": [0.1, 0.2, 0.3], "sampling_rate": 16000},
            "labels": [1, 2],
        },
        {
            "audio": {"array": [0.4, 0.5], "sampling_rate": 16000},
            "labels": [3],
        },
    ]


## Test Suite

def test_collator_call_structure(mock_processor, sample_features):
    """Test that the collator returns a dict with expected keys and types."""
    collator = DataCollatorCTCWithPadding(processor=mock_processor)
    batch = collator(sample_features)

    assert isinstance(batch, dict)
    assert "input_values" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert isinstance(batch["labels"], torch.Tensor)


def test_collator_processor_args(mock_processor, sample_features):
    """Test that audio processing receives correct input arrays and sampling rate."""
    collator = DataCollatorCTCWithPadding(processor=mock_processor, padding="longest")
    collator(sample_features)

    # Check processor call parameters
    mock_processor.assert_called_once_with(
        [[0.1, 0.2, 0.3], [0.4, 0.5]],
        sampling_rate=16000,
        padding="longest",
        return_tensors="pt",
    )


def test_collator_tokenizer_args(mock_processor, sample_features):
    """Test that tokenizer.pad receives label features correctly formatted."""
    collator = DataCollatorCTCWithPadding(processor=mock_processor, padding=True)
    collator(sample_features)

    expected_label_features = [
        {"input_ids": [1, 2]},
        {"input_ids": [3]},
    ]

    mock_processor.tokenizer.pad.assert_called_once_with(
        expected_label_features,
        padding=True,
        return_tensors="pt",
    )


def test_label_padding_masking_with_100(mock_processor, sample_features):
    """Test that attention mask 0s (padding positions) are properly converted to -100."""
    collator = DataCollatorCTCWithPadding(processor=mock_processor)
    batch = collator(sample_features)

    # In mock_pad:
    # input_ids      = [[1, 2], [3, 0]]
    # attention_mask = [[1, 1], [1, 0]]
    # Expected output: 0 at [1, 1] mapped to -100
    expected_labels = torch.tensor([[1, 2], [3, -100]])

    assert torch.equal(batch["labels"], expected_labels)


def test_custom_padding_parameter(mock_processor, sample_features):
    """Test passing custom padding strategy (e.g., 'max_length')."""
    collator = DataCollatorCTCWithPadding(processor=mock_processor, padding="max_length")
    collator(sample_features)

    assert mock_processor.call_args.kwargs["padding"] == "max_length"
    assert mock_processor.tokenizer.pad.call_args.kwargs["padding"] == "max_length"


def test_empty_batch_raises_index_error(mock_processor):
    """Test that calling collator on an empty list raises an IndexError."""
    collator = DataCollatorCTCWithPadding(processor=mock_processor)
    with pytest.raises(IndexError):
        collator([])