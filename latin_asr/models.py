
from transformers import Wav2Vec2ForCTC

from latin_asr.config import TrainingConfig


def build_model(
    config: TrainingConfig,
    pad_token_id: int,
    vocab_size: int,
    checkpoint: str,
    revision: str | None = None,
) -> Wav2Vec2ForCTC:
    """Instantiates Wav2Vec2ForCTC with configured regularization and SpecAugment hyperparameters."""
    model = Wav2Vec2ForCTC.from_pretrained(
        checkpoint,
        revision=revision,
        ignore_mismatched_sizes=True,
        ctc_loss_reduction="mean",
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        final_dropout=config.final_dropout,
        layerdrop=config.layerdrop,
        mask_time_prob=config.mask_time_prob,
        mask_time_length=config.mask_time_length,
        mask_feature_prob=config.mask_feature_prob,
        mask_feature_length=config.mask_feature_length,
    )

    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()

    return model