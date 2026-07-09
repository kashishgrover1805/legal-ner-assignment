"""
Step 3: standalone evaluation of a saved model checkpoint on the test split.
Useful to re-run evaluation without retraining, and to produce the
per-entity F1 table for the README.

Run:
    python src/evaluate_model.py --model_dir outputs/model/legalbert_indian/best
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
import evaluate

from src import config

seqeval = evaluate.load("seqeval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ds = load_from_disk(config.PROCESSED_DIR)["test"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    all_preds, all_labels = [], []
    id2label = model.config.id2label

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                true_pred, true_label = [], []
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l != -100:
                        true_pred.append(id2label[p])
                        true_label.append(id2label[l])
                all_preds.append(true_pred)
                all_labels.append(true_label)

    results = seqeval.compute(predictions=all_preds, references=all_labels)

    print("\n=== OVERALL (micro) ===")
    print(f"Precision: {results['overall_precision']:.4f}")
    print(f"Recall:    {results['overall_recall']:.4f}")
    print(f"F1:        {results['overall_f1']:.4f}")

    print("\n=== PER-ENTITY F1 ===")
    per_entity = {}
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"{key:15s}  P={val['precision']:.4f}  R={val['recall']:.4f}  F1={val['f1']:.4f}  support={val['number']}")
            per_entity[key] = val

    out = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "per_entity": per_entity,
    }
    out_path = os.path.join(config.METRICS_DIR, "test_eval_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
