"""
Step 5: measure CPU inference latency, model size, and peak RAM for a given
model directory (works for the plain PyTorch model, the ONNX FP32 model, or
the ONNX INT8-quantized model -- pass any of the three dirs).

Run:
    python src/benchmark.py --model_dir outputs/onnx_quantized --n_runs 50
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
from transformers import AutoTokenizer

from src import config

SAMPLE_TEXT = (
    "The petitioner, Ramesh Kumar, filed a writ petition before the "
    "Delhi High Court under Article 226 of the Constitution of India, "
    "challenging the order dated 12th March 2019 passed by the Trial Court "
    "in Suo Moto Writ Petition No. 45 of 2018, relying on the precedent "
    "set in State of Punjab v. Ajaib Singh (1953)."
)


def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if os.path.exists(os.path.join(model_dir, "model.onnx")) or os.path.exists(
        os.path.join(model_dir, "model_quantized.onnx")
    ):
        from optimum.onnxruntime import ORTModelForTokenClassification
        model = ORTModelForTokenClassification.from_pretrained(model_dir)
        backend = "onnxruntime"
    else:
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        backend = "pytorch"
    return tokenizer, model, backend


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    tokenizer, model, backend = load_model(args.model_dir)
    inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)

    process = psutil.Process(os.getpid())

    for _ in range(args.warmup):
        model(**inputs)

    mem_before = process.memory_info().rss / (1024 * 1024)
    peak_mem = mem_before
    latencies = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        model(**inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        current_mem = process.memory_info().rss / (1024 * 1024)
        peak_mem = max(peak_mem, current_mem)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    avg = sum(latencies) / len(latencies)

    print(f"\nBackend:        {backend}")
    print(f"Model dir:       {args.model_dir}")
    print(f"Model size:      {dir_size_mb(args.model_dir):.1f} MB")
    print(f"Avg latency:     {avg:.2f} ms")
    print(f"P50 latency:     {p50:.2f} ms")
    print(f"P95 latency:     {p95:.2f} ms")
    print(f"Peak RAM (RSS):  {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
