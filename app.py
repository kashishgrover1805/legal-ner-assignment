import streamlit as st

# -------------------- Page Configuration --------------------

st.set_page_config(
    page_title="Legal NER",
    page_icon="⚖️",
    layout="wide"
)

# -------------------- Imports --------------------

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification

from eval import merge_bio_spans

# -------------------- Hugging Face Model --------------------

MODEL_ID = "kashishgroverrr/legal-ner-onnx"

# -------------------- Load Model --------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = ORTModelForTokenClassification.from_pretrained(
        MODEL_ID
    )

    return tokenizer, model


tokenizer, model = load_model()

# -------------------- UI --------------------

st.title("⚖️ Legal Named Entity Recognition")

st.write(
    """
Detect legal entities from Indian court judgments using a fine-tuned
LegalBERT ONNX model.
"""
)

default_text = """The petitioner Ramesh Kumar filed a case before the Delhi High Court under Section 302 IPC on 12 March 2019."""

text = st.text_area(
    "Enter Legal Text",
    value=default_text,
    height=200
)

# -------------------- Prediction --------------------

if st.button("Predict Entities"):

    with st.spinner("Running inference..."):

        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True
        )

        offsets = encoding.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = model(**encoding)

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )[0].tolist()

        labels = [
            model.config.id2label[idx]
            for idx in predictions
        ]

        entities = merge_bio_spans(
            offsets,
            labels
        )

        # Add entity text
        for entity in entities:
            entity["text"] = text[entity["start"]:entity["end"]]

    if len(entities) == 0:

        st.warning("No entities detected.")

    else:

        st.success(f"Detected {len(entities)} entities")

        results = []

        for entity in entities:

            results.append(
                {
                    "Entity Type": entity["type"],
                    "Text": entity["text"],
                    "Start": entity["start"],
                    "End": entity["end"]
                }
            )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )