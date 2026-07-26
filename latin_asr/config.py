from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # --- Repos & Datasets ---
    hf_repo_id: str = "njand/wav2vec2-xls-r-latin"
    dataset_name: str = "njand/llpsi-speech-dataset"
    base_model: str = "facebook/wav2vec2-xls-r-300m"
    exp_prefix: str = "v4.0"
    
    # --- Pipeline & Optimization ---
    num_folds: int = 5
    target_fold_epochs: int = 16
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    bf16: bool = True
    fp16: bool = False

    # --- Model Regularization & Dropout ---
    attention_dropout: float = 0.15
    hidden_dropout: float = 0.15
    feat_proj_dropout: float = 0.10
    final_dropout: float = 0.20
    layerdrop: float = 0.10

    # --- SpecAugment ---
    mask_time_prob: float = 0.075
    mask_time_length: int = 10
    mask_feature_prob: float = 0.05
    mask_feature_length: int = 10

    # --- Freezing ---
    freeze_feature_encoder: bool = True