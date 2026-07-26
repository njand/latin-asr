from typing import Any


def get_processor(hf_repo_id: str, base_model: str, hf_token: str) -> Any:
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    try:
        print(f"Attempting to load full Wav2Vec2Processor from '{hf_repo_id}'...", flush=True)
        processor = Wav2Vec2Processor.from_pretrained(hf_repo_id, token=hf_token)
        processor.tokenizer.do_lower_case = False
        return processor
    except Exception:  # noqa: BLE001
        print("Fallback: Building processor with custom tokenizer...", flush=True)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            hf_repo_id, 
            token=hf_token, 
            do_lower_case=False  # Keep lowercase characters intact without converting to uppercase
        )
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model, token=hf_token)
        return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def load_and_prepare_dataset(processor: Any, dataset_name: str, hf_token: str) -> Any:
    """Loads raw dataset, tokenizes text labels, and casts audio features efficiently."""
    from datasets import Audio, load_dataset

    print(f"Loading raw dataset '{dataset_name}'...", flush=True)
    raw_ds = load_dataset(dataset_name, token=hf_token)
    ds = raw_ds.get("train", raw_ds)

    def prepare_example(batch):
        batch["labels"] = processor.tokenizer(batch["text_normalized"]).input_ids
        batch["length"] = len(batch["audio"]["array"])
        return batch

    print("Preparing text labels...", flush=True)
    cols_to_keep = {"audio", "fold", "labels", "text_normalized", "length"}
    cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    processed_ds = ds.map(
        prepare_example, 
        remove_columns=[c for c in cols_to_remove if c != "audio"],
        num_proc=4
    )

    return processed_ds