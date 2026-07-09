"""
Step 4: CPU inference optimization.

Converts the fine-tuned PyTorch model to ONNX, then applies dynamic INT8
quantization (weights only -- safe for CPU, no calibration data needed).
This is what eval.py and app.py actually load at inference time.

Run:
    python src/export_onnx.py --model_dir outputs/model/legalbert_indian/best
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from src import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="path to fine-tuned HF model (PyTorch)")
    args = parser.parse_args()

    print("Exporting to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(args.model_dir, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    ort_model.save_pretrained(config.ONNX_DIR)
    tokenizer.save_pretrained(config.ONNX_DIR)
    print(f"Saved ONNX model to {config.ONNX_DIR}")

    print("Applying dynamic INT8 quantization...")
    quantizer = ORTQuantizer.from_pretrained(config.ONNX_DIR)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=config.QUANT_DIR, quantization_config=qconfig)
    tokenizer.save_pretrained(config.QUANT_DIR)
    print(f"Saved quantized ONNX model to {config.QUANT_DIR}")

    # quick size report
    def dir_size_mb(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)

    print(f"\nFP32 ONNX size:      {dir_size_mb(config.ONNX_DIR):.1f} MB")
    print(f"INT8 quantized size: {dir_size_mb(config.QUANT_DIR):.1f} MB")


if __name__ == "__main__":
    main()
