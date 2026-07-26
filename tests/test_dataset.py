from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Tests for get_processor
# ---------------------------------------------------------------------------

@patch("transformers.Wav2Vec2Processor")
def test_get_processor_primary_path_success(mock_wav2vec2_processor):
    """Test successful loading of the full Wav2Vec2Processor from the primary repository."""
    from latin_asr.dataset import get_processor  # Replace 'your_module' with actual filename

    mock_instance = MagicMock()
    mock_wav2vec2_processor.from_pretrained.return_value = mock_instance

    result = get_processor(
        hf_repo_id="org/my-model",
        base_model="facebook/wav2vec2-base",
        hf_token="fake_token"
    )

    mock_wav2vec2_processor.from_pretrained.assert_called_once_with("org/my-model", token="fake_token")
    assert result == mock_instance
    assert mock_instance.tokenizer.do_lower_case is False


@patch("transformers.Wav2Vec2Processor")
@patch("transformers.Wav2Vec2FeatureExtractor")
@patch("transformers.Wav2Vec2CTCTokenizer")
def test_get_processor_fallback_path(
    mock_tokenizer_cls,
    mock_extractor_cls,
    mock_processor_cls
):
    """Test fallback path when primary Wav2Vec2Processor fails to load."""
    from latin_asr.dataset import get_processor

    # Force primary load to fail
    mock_processor_cls.from_pretrained.side_effect = Exception("Repo not found")

    mock_tokenizer_instance = MagicMock()
    mock_extractor_instance = MagicMock()
    mock_processor_instance = MagicMock()

    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    mock_extractor_cls.from_pretrained.return_value = mock_extractor_instance
    mock_processor_cls.return_value = mock_processor_instance

    result = get_processor(
        hf_repo_id="org/my-model",
        base_model="facebook/wav2vec2-base",
        hf_token="fake_token"
    )

    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "org/my-model", token="fake_token", do_lower_case=False
    )
    mock_extractor_cls.from_pretrained.assert_called_once_with(
        "facebook/wav2vec2-base", token="fake_token"
    )
    mock_processor_cls.assert_called_once_with(
        feature_extractor=mock_extractor_instance, tokenizer=mock_tokenizer_instance
    )
    assert result == mock_processor_instance


# ---------------------------------------------------------------------------
# Tests for load_and_prepare_dataset
# ---------------------------------------------------------------------------

@patch("datasets.load_dataset")
@patch("datasets.Audio")
def test_load_and_prepare_dataset_train_split(mock_audio_cls, mock_load_dataset):
    """Test dataset loading and mapping when 'train' split is present."""
    from latin_asr.dataset import load_and_prepare_dataset

    # Mock raw dataset with 'train' split
    mock_train_ds = MagicMock()
    mock_train_ds.column_names = ["audio", "text_normalized", "speaker_id", "extra_col"]
    mock_raw_ds = {"train": mock_train_ds}
    mock_load_dataset.return_value = mock_raw_ds

    # Setup cast_column and map chaining
    mock_casted_ds = MagicMock()
    mock_train_ds.cast_column.return_value = mock_casted_ds

    mock_final_ds = MagicMock()
    mock_casted_ds.map.return_value = mock_final_ds

    mock_processor = MagicMock()

    result = load_and_prepare_dataset(
        processor=mock_processor,
        dataset_name="my_dataset",
        hf_token="fake_token"
    )

    # Verify dataset load parameters
    mock_load_dataset.assert_called_once_with("my_dataset", token="fake_token")

    # Verify column removal logic:
    # cols_to_keep: {'audio', 'fold', 'labels', 'text_normalized', 'length'}
    # cols_to_remove from ['audio', 'text_normalized', 'speaker_id', 'extra_col'] -> ['speaker_id', 'extra_col']
    # remove_columns passed to map: [c for c in cols_to_remove if c != "audio"] -> ['speaker_id', 'extra_col']
    mock_casted_ds.map.assert_called_once()
    _, kwargs = mock_casted_ds.map.call_args

    assert set(kwargs["remove_columns"]) == {"speaker_id", "extra_col"}
    assert kwargs["num_proc"] == 4
    assert result == mock_final_ds


@patch("datasets.load_dataset")
@patch("datasets.Audio")
def test_load_and_prepare_dataset_fallback_split(mock_audio_cls, mock_load_dataset):
    """Test dataset loading when no 'train' key exists in raw_ds dict."""
    from latin_asr.dataset import load_and_prepare_dataset

    # Mock raw dataset without 'train' split
    mock_ds = MagicMock()
    mock_ds.column_names = ["audio", "text_normalized"]
    mock_ds.cast_column.return_value = mock_ds
    
    # Configure .get() to return default (fallback) when key is missing
    mock_ds.get.side_effect = lambda key, default=None: default
    mock_load_dataset.return_value = mock_ds

    mock_processor = MagicMock()

    load_and_prepare_dataset(
        processor=mock_processor,
        dataset_name="my_dataset",
        hf_token="fake_token"
    )

    mock_ds.cast_column.assert_called_once()


def test_prepare_example_closure_logic():
    """Directly test the prepare_example inner function logic."""
    from latin_asr.dataset import load_and_prepare_dataset

    mock_processor = MagicMock()
    mock_processor.tokenizer.return_value.input_ids = [10, 20, 30]

    batch = {
        "text_normalized": "hello world",
        "audio": {"array": [0.1, 0.2, 0.3, 0.4]}
    }

    # Intercept prepare_example during map execution
    captured_fn = None
    with patch("datasets.load_dataset") as mock_load_dataset, patch("datasets.Audio"):
        mock_ds = MagicMock()
        mock_ds.column_names = ["audio", "text_normalized"]
        mock_ds.cast_column.return_value = mock_ds
        
        # Ensure .get() falls back to mock_ds
        mock_ds.get.side_effect = lambda key, default=None: default
        mock_load_dataset.return_value = mock_ds

        def fake_map(fn, **kwargs):
            nonlocal captured_fn
            captured_fn = fn
            return mock_ds

        mock_ds.map.side_effect = fake_map

        load_and_prepare_dataset(mock_processor, "ds_name", "token")

    # Execute and test the captured prepare_example function
    processed_batch = captured_fn(batch)

    mock_processor.tokenizer.assert_called_once_with("hello world")
    assert processed_batch["labels"] == [10, 20, 30]
    assert processed_batch["length"] == 4