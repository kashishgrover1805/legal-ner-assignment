#!/usr/bin/env python
"""
CLI inference entrypoint (required deliverable).

Usage:
    python eval.py --text "The petitioner Ramesh Kumar filed a case before the Delhi High Court under Section 302 IPC on 12 March 2019."

    # or pipe text in:
    echo "some legal text" | python eval.py --stdin

Loads the CPU-optimized (ONNX INT8 quantized) model by default. Prints
detected entities with their grouped category and character span.
"""

import argparse
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config


def merge_bio_spans(tokens_with_offsets, labels):
    """Collapse token-level B-/I- predictions into entity spans with char offsets."""
    entities = []
    current = None

    for (start, end), label in zip(tokens_with_offsets, labels):
        if start == end:  # special token
            continue
        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        prefix, group = label.split("-", 1)
        if prefix == "B" or current is None or current["type"] != group:
            if current:
                entities.append(current)
            current = {"type": group, "start": start, "end": end}
        else:  # "I-" continuing the same group
            current["end"] = end

    if current:
        entities.append(current)
    return entities


def run_inference(text, model_dir):
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForTokenClassification.from_pretrained(model_dir)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label
    labels = [id2label[i] for i in pred_ids]

    spans = merge_bio_spans(offsets, labels)
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="single input string")
    parser.add_argument("--stdin", action="store_true", help="read input text from stdin")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=config.ONNX_DIR,
        help="path to model dir (default: FP32 ONNX CPU model -- see README Section 6/10 "
             "for why quantized INT8 is NOT used by default: it broke accuracy badly on "
             "short sentences in testing, even though it was smaller/faster)",
    )
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.error("provide --text '...' or --stdin")
        return

    entities = run_inference(text, args.model_dir)
    print(json.dumps({"text": text, "entities": entities}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
