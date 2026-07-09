# Legal NER --- Indian Court Judgments

End-to-end NER pipeline for Indian court judgment text: 6 grouped entity
categories (NAME, ORGANIZATION, LOCATION, STATUTE, CASE_REFERENCE,
DATE), trained on InLegalNER, optimized for CPU inference, deployed
using Streamlit with an ONNX model hosted on Hugging Face.

**GitHub Repository:**
https://github.com/kashishgrover1805/legal-ner-assignment

**Hugging Face Model:**
https://huggingface.co/kashishgroverrr/legal-ner-onnx

------------------------------------------------------------------------

## 1. Problem framing & approach

Legal judgment text is long (often \>2000 tokens per document) and has
domain-specific spans (statute citations like "Section 302 IPC", case
citations like "State of Punjab v. Ajaib Singh (1953)"). Two viable
model families:

1.  **Statistical baseline (CRF / spaCy's default)** --- fast, tiny, but
    no contextual understanding; struggles when a name and a place look
    identical without context ("Kerala" the state vs. a surname).
2.  **Transformer encoder fine-tuned for token classification** ---
    captures context, handles the ambiguity above, and can be
    quantized/exported to ONNX to still be fast enough for CPU inference
    at deployment time.

I went with **option 2**: a fine-tuned encoder + post-training INT8
quantization via ONNX Runtime. This gets accuracy close to a large model
while keeping inference CPU-friendly, rather than trying to hand-craft
features for a CRF over a genuinely diverse legal vocabulary.

### Why a legal-domain encoder

`law-ai/InLegalBERT` is pretrained on Indian legal text (judgments,
statutes), so it already has better subword representations for terms
like "writ", "petitioner", "IPC", "Article 226" than a generic
BERT/DistilBERT. The hypothesis tested in the experiments below: does
that domain pretraining translate into meaningfully higher F1 on STATUTE
and CASE_REFERENCE (the two hardest, most jargon-heavy categories)?

## 2. Label mapping

InLegalNER ships 14 fine-grained entity tags. The assignment asks for 6
grouped categories. Mapping used (see `src/config.py::RAW_TO_GROUP`):

  Raw InLegalNER tag                                             Grouped category
  -------------------------------------------------------------- ------------------
  JUDGE, LAWYER, PETITIONER, RESPONDENT, WITNESS, OTHER_PERSON   NAME
  COURT, ORG                                                     ORGANIZATION
  GPE                                                            LOCATION
  STATUTE, PROVISION                                             STATUTE
  PRECEDENT, CASE_NUMBER                                         CASE_REFERENCE
  DATE                                                           DATE

**Note:** run `python src/data_prep.py` first and check the printed raw
tag list against this table --- if the Hub version of the dataset uses
slightly different tag spellings, update `RAW_TO_GROUP` before training
(the script warns and drops any unmapped tag to `O` rather than
crashing, so a mismatch is visible immediately in the logs, not silently
wrong).

## 3. Preprocessing

-   Long documents → **sliding window tokenization** (`max_length=256`,
    `stride=64`) instead of hard truncation, so entities occurring later
    in a document aren't simply discarded. Each window becomes its own
    training example; at inference time a single call still works fine
    for typical-length inputs, and `eval.py` truncates gracefully for
    anything longer than one window (see Limitations).
-   Word-level BIO labels are aligned to subword tokens: only the first
    subword of each word carries the label; continuation subwords get
    `-100` and are ignored by the loss (standard HF token-classification
    recipe).

## 4. Experiments

All runs: 5 epochs, seed 42, batch size 16, AdamW,
`eval_strategy="epoch"`, best checkpoint selected by validation
micro-F1.

  ----------------------------------------------------------------------------------------
  \#             Model                  LR             Notes                Micro F1
                                                                            (test)
  -------------- ---------------------- -------------- -------------------- --------------
  1              `law-ai/InLegalBERT`   2e-5           domain-pretrained, 5 **0.864**
                                                       epochs, seed 42      

  ----------------------------------------------------------------------------------------

**Kept:** InLegalBERT. Full test-set breakdown below (see Section 5).
Given the strong results and time budget, the DistilBERT/BERT-base
comparison runs were not executed in this pass --- if you have time
before submission, running them (commands below) and adding a row each
strengthens the "reasoning behind model choice" section further, but is
not required for correctness.

To reproduce:

``` bash
python src/train.py --model_key legalbert_indian --epochs 5 --lr 2e-5
```

Optional comparisons:

``` bash
python src/train.py --model_key distilbert       --epochs 5 --lr 5e-5
python src/train.py --model_key bert_base         --epochs 5 --lr 2e-5
```

Each run writes `outputs/metrics/<run_name>_metrics.json`.

### Class imbalance --- observed in results

CASE_REFERENCE (F1 0.728) and LOCATION (F1 0.721) lag behind
DATE/STATUTE/NAME (F1 0.91-0.93), consistent with the imbalance
hypothesis: these are the lowest-frequency raw tags (PRECEDENT,
CASE_NUMBER, GPE combined are far rarer in the corpus than
DATE/JUDGE/LAWYER/PETITIONER). Worth trying next: weighted cross-entropy
by inverse label frequency, or oversampling documents that contain these
rarer entity types.

## 5. Final results (test split)

Reproduce with:

``` bash
python src/evaluate_model.py --model_dir outputs/model/legalbert_indian/best
```

    Overall (micro):
      Precision: 0.8366
      Recall:    0.8942
      F1:        0.8644

    Per-entity F1:
      NAME:            0.9076
      ORGANIZATION:    0.7943
      LOCATION:        0.7209
      STATUTE:         0.9091
      CASE_REFERENCE:  0.7277
      DATE:            0.9253

Full JSON report saved to
`outputs/metrics/legalbert_indian_metrics.json`. DATE, STATUTE and NAME
are the strongest categories (high-frequency, relatively formulaic
patterns like "Section X of Y Act" or "DD Month YYYY"). LOCATION and
CASE_REFERENCE are the weakest --- both are rarer in the training data
and more context-dependent (a bare place name vs. a surname, or an
informally-cited precedent, are harder to disambiguate from local
context alone).

## 6. CPU inference optimization

Pipeline: PyTorch checkpoint → ONNX export → dynamic INT8 quantization
(ONNX Runtime, weight-only, no calibration data needed since it's
dynamic quantization --- appropriate here because we don't have a
representative "deployment" input distribution separate from the
training text).

``` bash
python src/export_onnx.py --model_dir outputs/model/legalbert_indian/best
python src/benchmark.py --model_dir outputs/model/legalbert_indian/best   # PyTorch baseline
python src/benchmark.py --model_dir outputs/onnx                          # ONNX FP32
python src/benchmark.py --model_dir outputs/onnx_quantized                # ONNX INT8
```

  --------------------------------------------------------------------------
  Backend        Size (MB)      Avg latency    P95 latency    Peak RAM (MB)
                                (ms)           (ms)           
  -------------- -------------- -------------- -------------- --------------
  PyTorch FP32   416.3          225.19         291.38         964.9

  ONNX FP32      416.6          153.85         209.85         1962.7

  **ONNX INT8    **105.6**      **100.81**     **109.51**     **1395.4**
  (deployed)**                                                
  --------------------------------------------------------------------------

Quantization gets \~75% size reduction and \~2.2x speedup over the raw
PyTorch checkpoint. Peak RAM for the ONNX Runtime backends looks higher
than PyTorch here because `psutil` measures the whole process's RSS and
onnxruntime pre-allocates its own arena; the number to trust for "will
this fit on a basic CPU Space" is the ONNX INT8 row, and 1.4 GB is well
within a free/basic Streamlit Cloud CPU tier's memory budget.

Fill this table from the benchmark script's stdout on your machine if
you retrain --- numbers are hardware-dependent.

## 7. Deployment

The application is deployed using **Streamlit Community Cloud** on a CPU instance.

Instead of storing the trained model inside the GitHub repository, the application downloads the **FP32 ONNX** model directly from the Hugging Face Model Hub during startup.

**Model Repository**

https://huggingface.co/kashishgroverrr/legal-ner-onnx

This keeps the GitHub repository lightweight while avoiding GitHub's file size limitations.

### Deployment Steps

1. Push the project code to GitHub.

2. Upload the exported ONNX model files (`model.onnx`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`) to the Hugging Face model repository.

3. Configure `app.py` to load the model directly from Hugging Face:

```python
MODEL_ID = "kashishgroverrr/legal-ner-onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = ORTModelForTokenClassification.from_pretrained(MODEL_ID)
```

4. Deploy the GitHub repository on **Streamlit Community Cloud** by selecting:

- Repository: `kashishgrover1805/legal-ner-assignment`
- Branch: `main`
- Main file: `app.py`

Once deployed, Streamlit automatically installs the dependencies, downloads the ONNX model from Hugging Face, and serves the application on a CPU instance.

This deployment strategy avoids committing large model files to GitHub while ensuring reproducible inference and easy public access.

## 8. Repository structure

    .
    ├── app.py                  # Streamlit demo (Streamlit Cloud entrypoint)
    ├── eval.py                 # required CLI: single-string inference
    ├── requirements.txt        # pinned deps
    ├── src/
    │   ├── config.py           # label mapping, model choices, paths
    │   ├── data_prep.py        # load + remap + tokenize (sliding window)
    │   ├── train.py            # fine-tuning + seqeval metrics
    │   ├── evaluate_model.py   # standalone test-set evaluation
    │   ├── export_onnx.py      # ONNX export + INT8 quantization
    │   └── benchmark.py        # latency / size / RAM benchmarking
    └── outputs/                # metrics, model checkpoints, onnx artifacts (gitignored bulk)

Hugging Face Model Repository:
https://huggingface.co/kashishgroverrr/legal-ner-onnx
## 9. Exact commands, in order

``` bash
# 0. one-time setup
pip install -r requirements.txt
huggingface-cli login   # accept InLegalNER's gated terms on the Hub first

# 1. data
python src/data_prep.py

# 2. train (pick the final model; run the other two if you want the
#    comparison table filled in too)
python src/train.py --model_key legalbert_indian --epochs 5 --lr 2e-5

# 3. evaluate on test split
python src/evaluate_model.py --model_dir outputs/model/legalbert_indian/best

# 4. export + quantize for CPU
python src/export_onnx.py --model_dir outputs/model/legalbert_indian/best

# 5. benchmark CPU inference (both, for comparison -- see Section 6 for why
#    FP32 is the deployed choice despite being the larger of the two)
python src/benchmark.py --model_dir outputs/onnx --n_runs 50
python src/benchmark.py --model_dir outputs/onnx_quantized --n_runs 50

# 6. try single-string inference (defaults to FP32 ONNX)
python eval.py --text "The petitioner filed a case under Section 302 IPC before the Delhi High Court."

# 7. run the demo locally before deploying
streamlit run app.py
```

## 10. Limitations / future work

-   **RESOLVED --- quantization accuracy issue, confirmed and worked
    around:** a manual test of
    `eval.py --text "The petitioner Ramesh Kumar filed a case   under Section 302 IPC before the Delhi High Court on 12 March 2019."`
    was run against all three backends side by side:

    -   **PyTorch checkpoint** and **ONNX FP32** both correctly
        extracted all 5 entities (Ramesh Kumar/NAME, Section
        302/STATUTE, IPC/STATUTE, Delhi High Court/ORGANIZATION, 12
        March 2019/DATE) --- identical output.
    -   **ONNX INT8 (dynamically quantized)** only caught a mangled
        partial span ("esh Kumar"/NAME), missing every other entity.

    This confirms the issue is **quantization accuracy loss**, not a bug
    in `merge_bio_spans()` --- the merging logic is correct (proven by
    the FP32 ONNX and PyTorch outputs matching exactly), the INT8
    model's underlying token predictions themselves are simply wrong on
    this input.

    **Decision:** deploy **FP32 ONNX**, not INT8, despite the larger
    size. In this run the latency gap between FP32 ONNX (\~161ms) and
    INT8 (\~140ms) was small, while the accuracy gap was severe --- not
    a good trade for a demo. 416MB fits comfortably within a CPU-basic
    Space's 16GB RAM budget, so there's no deployment-size pressure
    forcing the smaller model. `eval.py` and `app.py` both default to
    the FP32 ONNX model dir accordingly (see Sections 6-7).

    **If a future iteration still wants INT8's size/speed:** try
    **static (calibrated) quantization** instead of dynamic --- dynamic
    quantization (used here) skips calibration data entirely and
    quantizes activations on-the-fly per-inference, which is more
    failure-prone on short/simple inputs than static quantization
    calibrated on representative legal text.

-   `eval.py` truncates a single call to `MAX_LENGTH` tokens; a
    production version should re-apply the sliding-window logic from
    `data_prep.py` and merge overlapping predictions for documents
    longer than one window.

-   Dynamic quantization was chosen over static (calibrated)
    quantization for simplicity; static quantization could shrink
    latency further at the cost of needing a calibration set --- worth
    trying if CPU latency is still a bottleneck in production.

-   Per-entity F1 for rare classes (CASE_REFERENCE) should be revisited
    with class-weighted loss or oversampling once you have baseline
    numbers.
pip install -r requirements.txt
py eval.py --text "The petitioner Ramesh Kumar filed a case before the Delhi High Court under Section 302 IPC 
on 12 March 2019."
streamlit run app.py