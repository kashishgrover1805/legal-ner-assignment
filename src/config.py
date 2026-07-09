"""
Central configuration: dataset id, label mapping (14 -> 6 categories),
model checkpoints, and shared paths.

IMPORTANT: The InLegalNER dataset is gated on Hugging Face. Before running
anything, log in on the machine that will actually download data:

    huggingface-cli login

and make sure you have clicked "Agree" on the dataset page.

The exact HF repo id / column names can change over time or differ from what
is documented here from memory. The FIRST thing train.py's data prep step
does is print dataset.features and a sample row -- always check that against
this file before trusting the mapping below. If the raw tag names differ,
edit RAW_TO_GROUP accordingly; nothing else needs to change.
"""

import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Update this if the dataset resolves under a different HF namespace after
# you search for it in the Hub UI (search "InLegalNER").
DATASET_ID = "opennyaiorg/InLegalNER"

# ---------------------------------------------------------------------------
# Final 6 categories required by the assignment
# ---------------------------------------------------------------------------
GROUPS = ["NAME", "ORGANIZATION", "LOCATION", "STATUTE", "CASE_REFERENCE", "DATE"]

# Mapping from the InLegalNER raw entity tags (as published in the
# Kalamkar et al. 2022 "Named Entity Recognition in Indian court judgments"
# annotation scheme) to our 6 grouped categories. Raw tags come already
# split by legal-NER convention; we collapse person/institution variants.
RAW_TO_GROUP = {
    "JUDGE": "NAME",
    "LAWYER": "NAME",
    "PETITIONER": "NAME",
    "RESPONDENT": "NAME",
    "WITNESS": "NAME",
    "OTHER_PERSON": "NAME",
    "COURT": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "STATUTE": "STATUTE",
    "PROVISION": "STATUTE",
    "PRECEDENT": "CASE_REFERENCE",
    "CASE_NUMBER": "CASE_REFERENCE",
    "DATE": "DATE",
}

# Build final BIO label list: O + B-/I- for each group, deterministic order.
LABEL_LIST = ["O"]
for g in GROUPS:
    LABEL_LIST.append(f"B-{g}")
    LABEL_LIST.append(f"I-{g}")

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Models tried (see README "Experiments" table)
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = {
    "distilbert": "distilbert-base-uncased",
    "legalbert_indian": "law-ai/InLegalBERT",
    "bert_base": "bert-base-uncased",
}

# Final model chosen for training/deployment (edit after running experiments)
ACTIVE_MODEL_KEY = "legalbert_indian"
ACTIVE_MODEL_NAME = MODEL_CANDIDATES[ACTIVE_MODEL_KEY]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT, "outputs", "model")
ONNX_DIR = os.path.join(ROOT, "outputs", "onnx")
QUANT_DIR = os.path.join(ROOT, "outputs", "onnx_quantized")
METRICS_DIR = os.path.join(ROOT, "outputs", "metrics")

for d in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, ONNX_DIR, QUANT_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Tokenization / windowing
# ---------------------------------------------------------------------------
MAX_LENGTH = 256
STRIDE = 64  # overlap between windows so entities at boundaries aren't cut

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
SEED = 42
