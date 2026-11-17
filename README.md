# Support Ticket Router (SWE015 Homework, Use Case 2)

A web application that takes a customer message, cleans it, predicts the intent with four models and routes it to a team. Low-confidence and out-of-scope messages go to a human agent; greetings and small talk get a rule-based reply.

## How it works

1. **Preprocessing** (`src/preprocess.py`): regex normalization, model-code extraction (e.g. `DW-6534`), minimum-edit-distance correction of misspelled product names.
2. **Small talk** (`src/smalltalk.py`): ELIZA-style regex rules.
3. **Four models** predict one of 150 intents or `oos` (out-of-scope):

| Model | Area | Precision |
|---|---|---|
| tf-idf + Logistic Regression | B (classical NLP) | fp32 |
| Feedforward network on averaged GloVe vectors, trained with SGD | A (neural network) | fp32 |
| BERT-base, fine-tuned | C (Transformer) | fp32 |
| DistilBERT student, distilled from BERT, dynamic quantization | D (distillation + quantization) | int8 |

4. **Routing** (`src/teams.py`, `src/pipeline.py`): the **int8 student decides**. If the prediction is `oos` or its confidence is below a threshold τ, the message goes to a human agent; otherwise the intent is mapped to a team. τ is chosen per model on the validation set.

## Results (test set: 4,500 in-scope + 1,000 out-of-scope messages)

| Model | In-scope accuracy | OOS recall | Team accuracy | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|
| tf-idf + LogReg | 0.7991 | 0.918 | 0.8178 | 39.4 | 0.4 |
| FFN (GloVe avg) | 0.7751 | 0.867 | 0.7924 | 0.9 | 0.1 |
| BERT-base (fp32) | 0.9349 | 0.899 | 0.9464 | 438.5 | 48.2 |
| DistilBERT student (fp32) | 0.8929 | 0.907 | 0.9007 | 268.3 | 28.9 |
| DistilBERT student (int8) | 0.9078 | 0.892 | 0.9191 | 138.8 | 9.4 |

In-scope accuracy and OOS recall use the threshold τ chosen on the validation set. Latency is the mean of 200 single-message runs (preprocessing included) on an Apple M4, macOS 15, 4 threads, qnnpack engine. Full numbers: `results/results.csv`.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. The first run downloads the model weights from the Hugging Face Hub (about 750 MB) and the GloVe vectors (about 130 MB), so it takes a few minutes. Tested with Python 3.13 on macOS (Apple Silicon).

## Model weights (not stored in this repository)

- BERT-base teacher: https://huggingface.co/goncayldz/clinc150-bert-teacher
- DistilBERT student: https://huggingface.co/goncayldz/clinc150-distilbert-student
- tf-idf + LogReg and FFN: https://huggingface.co/goncayldz/clinc150-classical-models

The int8 model is not uploaded: the app quantizes the fp32 student at start-up.

## Reproducing the training

```bash
python src/data.py                # download CLINC150 ("plus") to data/raw
python src/baseline.py            # tf-idf + LogReg
python src/embed.py               # GloVe sentence vectors
python src/ffn.py                 # feedforward network (SGD)
# BERT fine-tuning and distillation: notebooks/bert_finetune_and_distill.ipynb (Google Colab, T4 GPU)
python src/quantize_student.py    # int8 quantization, size and latency
python src/evaluate.py            # test-set evaluation -> results/results.csv
```

## Limitations

- The data is English only (CLINC150). Turkish messages are not understood and are routed to a human agent with low confidence.
- Intents are grouped into teams by CLINC150 domain. The domains are not appliance-specific, so the team names are illustrative.
- Thresholds τ were selected on a validation set with only 100 out-of-scope messages; OOS recall dropped on the larger test set.
- `torch.ao.quantization` is announced as deprecated; `requirements.txt` pins `torch==2.15.0`, where it works.
- Single training run (one seed); small differences between models should not be over-interpreted.

## Data and references

- CLINC150: Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", EMNLP-IJCNLP 2019 (Hugging Face: `clinc/clinc_oos`, config `plus`).
- GloVe: Pennington et al., 2014 (`glove-wiki-gigaword-100` via gensim).
- BERT: Devlin et al., 2019. DistilBERT: Sanh et al., 2019.

## AI usage

Claude (Anthropic) was used as an assistant for writing and debugging code, explaining methods, and drafting this README and the presentation slides. I ran all training and evaluation myself, and all numbers were measured by me.