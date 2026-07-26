# /// script
# dependencies = [
#     "optimum[onnxruntime]>=2.1.0",
#     "onnx>=1.22.0",
#     "transformers<4.58.0",
# ]
# ///

"""
Export, quantize, benchmark, and publish models to ONNX
format for accelerated CPU inference.
"""

import logging
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from huggingface_hub import HfApi
from optimum.onnxruntime import ORTModelForCTC, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoProcessor

# Configure visual logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_onnx")

# Mapping CPU architecture profiles to Optimum configs
CONFIG_MAP = {
    "avx2": AutoQuantizationConfig.avx2,
    "avx512": AutoQuantizationConfig.avx512,
    "arm64": AutoQuantizationConfig.arm64,
}


def export_fp32(model_id: str, output_dir: Path) -> Path:
    """Exports PyTorch Wav2Vec2 weights and processor configs to ONNX FP32."""
    logger.info("Exporting PyTorch model '%s' to ONNX FP32...", model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights & preprocessor configs
    model = ORTModelForCTC.from_pretrained(model_id, export=True)
    model.save_pretrained(output_dir)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)

    fp32_path = output_dir / "model.onnx"
    size_mb = fp32_path.stat().st_size / 1e6
    logger.info("Saved ONNX FP32 model: %s (%.1f MB)", fp32_path, size_mb)
    return fp32_path


def quantize_int8(output_dir: Path, arch: str = "avx2") -> Path:
    """Applies dynamic INT8 quantization to the exported ONNX model."""
    logger.info("Quantizing ONNX model to INT8 (profile: %s)...", arch)

    config_factory = CONFIG_MAP.get(arch.lower(), AutoQuantizationConfig.avx2)
    # Restrict quantization operators to MatMul to prevent errors on dynamic weights
    # in Wav2Vec2's convolutional feature extractor and positional embeddings.
    qconfig = config_factory(
        is_static=False,
        per_channel=False,
        operators_to_quantize=["MatMul"],
    )

    quantizer = ORTQuantizer.from_pretrained(output_dir)
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=qconfig,
        file_suffix="quantized",
    )

    int8_path = output_dir / "model_quantized.onnx"
    size_mb = int8_path.stat().st_size / 1e6
    logger.info("Saved INT8 model: %s (%.1f MB)", int8_path, size_mb)
    return int8_path


def benchmark(output_dir: Path, filename: str, duration_sec: int = 5) -> float:
    """Measures inference latency on CPU using dummy audio input."""
    processor = AutoProcessor.from_pretrained(output_dir)
    model = ORTModelForCTC.from_pretrained(output_dir, file_name=filename)

    dummy_audio = torch.randn(1, 16000 * duration_sec)
    inputs = processor(dummy_audio.squeeze(0), sampling_rate=16000, return_tensors="pt")

    # Warmup pass
    _ = model(**inputs)

    # Timed pass
    start = time.perf_counter()
    outputs = model(**inputs)
    latency_ms = (time.perf_counter() - start) * 1000

    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    _ = processor.batch_decode(predicted_ids)[0]

    logger.info("Latency (%s): %.2f ms", filename, latency_ms)
    return latency_ms


def upload_to_hub(output_dir: Path, repo_id: str, token: str | None = None) -> None:
    """Uploads exported .onnx files and configs to the target Hugging Face Hub repository."""
    logger.info("Uploading ONNX models and configuration to Hugging Face Hub repo '%s'...", repo_id)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["*.onnx", "*.json"],
    )
    logger.info("Uploaded to https://huggingface.co/%s", repo_id)


def parse_args():
    parser = ArgumentParser(description="Export and quantize Wav2Vec2 models to ONNX for CPU inference.")
    parser.add_argument(
        "--model-id",
        default="njand/wav2vec2-xls-r-latin",
        help="Hugging Face repo ID or local checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./onnx_export"),
        help="Output directory for exported ONNX files.",
    )
    parser.add_argument(
        "--arch",
        choices=["avx2", "avx512", "arm64"],
        default="avx2",
        help="Target CPU instruction set for quantization.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Publish exported ONNX models to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face API write token.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Pipeline execution
    export_fp32(args.model_id, args.output_dir)
    quantize_int8(args.output_dir, arch=args.arch)

    # Sanity checks
    benchmark(args.output_dir, "model.onnx")
    benchmark(args.output_dir, "model_quantized.onnx")

    # Publishing
    if args.push_to_hub:
        upload_to_hub(args.output_dir, args.model_id, token=args.hf_token)


if __name__ == "__main__":
    main()