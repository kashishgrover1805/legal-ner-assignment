"""
Step 1 of the pipeline.

The InLegalNER dataset (as it actually ships on the Hub) is in Label-Studio
export format:

    {
      "data": {"text": "<raw judgment text>"},
      "annotations": [
        {"result": [
            {"value": {"start": 90, "end": 103, "labels": ["ORG"]}},
            ...
        ]}
      ]
    }

i.e. raw text + character-level entity spans, NOT pre-tokenized
words/BIO-tags. So the approach here is:

1. Pull (text, [(start, end, raw_label), ...]) out of each example.
2. Map each raw_label (e.g. "ORG", "PETITIONER") to one of our 6 grouped
   categories via config.RAW_TO_GROUP.
3. Tokenize the raw text directly with a sliding window
   (return_offsets_mapping=True lets us go straight from character spans to
   subword-level BIO labels, no separate word-alignment step needed since
   we never had word-level tokens to begin with).

Run:
    python src/data_prep.py
"""

import json
import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

from src import config


def load_raw():
    print(f"Loading dataset: {config.DATASET_ID}")
    ds = load_dataset(config.DATASET_ID)
    print("\n=== RAW DATASET SCHEMA ===")
    print(ds)
    return ds


def extract_text_and_entities(example):
    """Pull raw text + list of (start, end, raw_label) from one Label-Studio row."""
    text = example["data"]["text"]
    entities = []
    for ann in example["annotations"]:
        for res in ann["result"]:
            val = res["value"]
            labels = val.get("labels", [])
            if not labels:
                continue
            raw_label = labels[0].upper()
            entities.append((val["start"], val["end"], raw_label))
    entities.sort(key=lambda x: x[0])
    return text, entities


def scan_raw_label_set(ds):
    """Print every raw label actually present so RAW_TO_GROUP can be checked."""
    counts = Counter()
    for split in ds:
        for ex in ds[split]:
            for ann in ex["annotations"]:
                for res in ann["result"]:
                    for lab in res["value"].get("labels", []):
                        counts[lab.upper()] += 1
    print("\n=== RAW LABEL COUNTS (across all splits) ===")
    for lab, cnt in counts.most_common():
        mapped = config.RAW_TO_GROUP.get(lab, "**UNMAPPED -> will become O**")
        print(f"  {lab:15s} count={cnt:6d}  -> {mapped}")
    return counts


def char_spans_to_bio_labels(offsets, entities_grouped):
    """
    offsets: list of (char_start, char_end) per token, from the tokenizer,
             with (0, 0) for special/padding tokens.
    entities_grouped: list of (start, end, group_name) already mapped to one
                       of the 6 categories.
    Returns list of label strings, one per token, e.g. ["O", "B-NAME", "I-NAME", ...]
    """
    labels = ["O"] * len(offsets)
    for start, end, group in entities_grouped:
        first = True
        for i, (tok_s, tok_e) in enumerate(offsets):
            if tok_s == tok_e:
                continue  # special/padding token, never labeled here
            if tok_e <= start or tok_s >= end:
                continue  # no overlap with this entity
            labels[i] = f"B-{group}" if first else f"I-{group}"
            first = False
    return labels


def build_split(raw_split, tokenizer):
    """
    For every document: extract char-span entities, map to grouped labels,
    tokenize with a sliding window, convert offsets -> BIO label ids per
    window. Returns a HF Dataset of input_ids/attention_mask/labels.
    """
    out_input_ids, out_attention_mask, out_labels = [], [], []

    for example in raw_split:
        text, raw_entities = extract_text_and_entities(example)
        if not text.strip():
            continue

        grouped_entities = []
        for start, end, raw_label in raw_entities:
            group = config.RAW_TO_GROUP.get(raw_label)
            if group is None:
                continue  # unmapped tag -> treated as O, i.e. simply dropped
            grouped_entities.append((start, end, group))

        enc = tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_LENGTH,
            stride=config.STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        num_windows = len(enc["input_ids"])
        for w in range(num_windows):
            offsets = enc["offset_mapping"][w]
            label_strs = char_spans_to_bio_labels(offsets, grouped_entities)
            label_ids = [
                -100 if offsets[i] == (0, 0) else config.LABEL2ID[label_strs[i]]
                for i in range(len(offsets))
            ]
            out_input_ids.append(enc["input_ids"][w])
            out_attention_mask.append(enc["attention_mask"][w])
            out_labels.append(label_ids)

    return Dataset.from_dict(
        {
            "input_ids": out_input_ids,
            "attention_mask": out_attention_mask,
            "labels": out_labels,
        }
    )


def main():
    ds = load_raw()
    scan_raw_label_set(ds)

    tokenizer = AutoTokenizer.from_pretrained(config.ACTIVE_MODEL_NAME)

    print("\nBuilding tokenized windows for each split (this can take a couple minutes)...")
    processed = DatasetDict()
    for split_name in ds.keys():
        print(f"  processing split: {split_name} ({len(ds[split_name])} raw documents)")
        processed[split_name] = build_split(ds[split_name], tokenizer)
        print(f"    -> {len(processed[split_name])} tokenized windows")

    processed.save_to_disk(config.PROCESSED_DIR)
    print(f"\nSaved processed dataset to {config.PROCESSED_DIR}")

    with open(os.path.join(config.PROCESSED_DIR, "label_list.json"), "w") as f:
        json.dump(config.LABEL_LIST, f, indent=2)

    print("\nLabel list (grouped, 6 categories + O):")
    print(config.LABEL_LIST)


if __name__ == "__main__":
    main()
