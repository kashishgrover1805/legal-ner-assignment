"""
Step 2: fine-tune a token-classification model on the processed dataset.

Run (after data_prep.py):
    python src/train.py --model_key legalbert_indian --epochs 5 --lr 2e-5

Try other candidates for comparison (see README experiment log):
    python src/train.py --model_key distilbert --epochs 5 --lr 5e-5
    python src/train.py --model_key bert_base --epochs 5 --lr 2e-5
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

from src import config

seqeval = evaluate.load("seqeval")


def compute_metrics_builder(id2label):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        flat = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        # per-entity F1
        for key, val in results.items():
            if isinstance(val, dict):
                flat[f"f1_{key}"] = val["f1"]
                flat[f"precision_{key}"] = val["precision"]
                flat[f"recall_{key}"] = val["recall"]
        return flat

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=config.ACTIVE_MODEL_KEY,
                         choices=list(config.MODEL_CANDIDATES.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    set_seed(config.SEED)

    model_name = config.MODEL_CANDIDATES[args.model_key]
    run_name = args.run_name or args.model_key
    output_dir = os.path.join(config.MODEL_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading processed dataset from {config.PROCESSED_DIR}")
    ds = load_from_disk(config.PROCESSED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(config.LABEL_LIST),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
        fp16=False,  # set True if training on a CUDA GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(config.ID2LABEL),
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\n=== FINAL TEST METRICS ===")
    print(json.dumps(metrics, indent=2))

    metrics_path = os.path.join(config.METRICS_DIR, f"{run_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save the best model + tokenizer for downstream export/inference
    final_dir = os.path.join(output_dir, "best")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved best model to {final_dir}")


if __name__ == "__main__":
    main()
