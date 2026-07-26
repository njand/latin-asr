from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that dynamically pads audio inputs and text label sequences."""

    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        audio_arrays = [feature["audio"]["array"] for feature in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]

        batch = self.processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding=self.padding,
            return_tensors="pt",
        )
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad dynamic text batch
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Mask padding tokens for CTC loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels

        return batch