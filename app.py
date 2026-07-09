"""
Hugging Face Spaces entrypoint (Gradio SDK, CPU basic hardware).

Loads the FP32 ONNX model (see src/export_onnx.py) rather than the
INT8-quantized one. Testing showed INT8 quantization badly hurt accuracy on
short sentences (see README Section 10) even though it was smaller/faster,
so FP32 ONNX is the deployed choice -- 416MB comfortably fits a CPU basic
Space's 16GB RAM, and FP32 ONNX (~160ms) is already fast enough without
quantization's accuracy cost. Expects the model files to be present in
./model (copy outputs/onnx/* here before pushing the Space).
"""

import os
import gradio as gr
import torch
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

from eval import merge_bio_spans

MODEL_DIR = os.environ.get("MODEL_DIR", "./model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ORTModelForTokenClassification.from_pretrained(MODEL_DIR)

LABEL_COLORS = {
    "NAME": "#f9c74f",
    "ORGANIZATION": "#90be6d",
    "LOCATION": "#43aa8b",
    "STATUTE": "#577590",
    "CASE_REFERENCE": "#f3722c",
    "DATE": "#f94144",
}

EXAMPLE = (
    "The petitioner, Ramesh Kumar, filed a writ petition before the Delhi "
    "High Court under Article 226 of the Constitution of India, challenging "
    "the order dated 12th March 2019 passed in Suo Moto Writ Petition No. 45 "
    "of 2018, relying on the precedent in State of Punjab v. Ajaib Singh."
)


def predict(text):
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, return_offsets_mapping=True
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label
    labels = [id2label[i] for i in pred_ids]

    spans = merge_bio_spans(offsets, labels)

    highlighted = []
    cursor = 0
    for s in sorted(spans, key=lambda x: x["start"]):
        if s["start"] > cursor:
            highlighted.append((text[cursor:s["start"]], None))
        highlighted.append((text[s["start"]:s["end"]], s["type"]))
        cursor = s["end"]
    if cursor < len(text):
        highlighted.append((text[cursor:], None))

    return highlighted


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, label="Legal text", value=EXAMPLE),
    outputs=gr.HighlightedText(label="Detected entities", color_map=LABEL_COLORS),
    title="Legal NER — Indian Court Judgments",
    description=(
        "Extracts NAME, ORGANIZATION, LOCATION, STATUTE, CASE_REFERENCE and "
        "DATE entities from Indian court judgment text. Runs on CPU with an "
        "INT8-quantized ONNX model for fast inference."
    ),
    examples=[[EXAMPLE]],
)

if __name__ == "__main__":
    demo.launch()
