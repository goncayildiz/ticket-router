import warnings
warnings.filterwarnings("ignore")

import copy
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from embed import featurize, load_glove
from ffn import make_model
from metrics import final_predictions, in_scope_accuracy, oos_recall
from preprocess import normalize
from smalltalk import match_small_talk
from teams import SMALL_TALK_TEAM, team_of

TEACHER = "goncayldz/clinc150-bert-teacher"
STUDENT = "goncayldz/clinc150-distilbert-student"
MAX_LEN = 40

torch.manual_seed(0)
if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"

names = json.load(open("data/raw/label_names.json"))
OOS = names.index("oos")
te = pd.read_csv("data/raw/test.csv")
texts = te.text.tolist()
y = te.intent.values
TEAM = np.array([team_of(n) for n in names])

try:
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
except Exception:
    cpu = platform.processor()
hw = f"{cpu} | {platform.platform()} | torch {torch.__version__} | threads {torch.get_num_threads()} | engine {torch.backends.quantized.engine}"
print("hardware:", hw)
os.makedirs("results", exist_ok=True)
open("results/hardware.txt", "w").write(hw + "\n")


def size_mb_module(m):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(m.state_dict(), path)
    s = os.path.getsize(path) / 1e6
    os.remove(path)
    return s


def hf_batch_fn(model, tok):
    @torch.inference_mode()
    def fn(batch_texts, bs=64):
        out = []
        for i in range(0, len(batch_texts), bs):
            enc = tok([normalize(t) for t in batch_texts[i:i + bs]], padding=True,
                      truncation=True, max_length=MAX_LEN, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            out.append(torch.softmax(logits.float(), dim=-1).numpy())
        return np.concatenate(out)
    return fn


def latency(fn, n=200, warm=20):
    pool = texts[:20]
    for i in range(warm):
        fn([pool[i % 20]])
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn([pool[i % 20]])
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), statistics.median(ts), float(np.percentile(ts, 95))


models = []

# 1) tf-idf + Logistic Regression (Area B)
lr = joblib.load("models/tfidf_lr.joblib")
models.append({
    "name": "tfidf + LogReg", "area": "B",
    "fn": lambda bt: lr["pipe"].predict_proba([normalize(t) for t in bt]),
    "tau": lr["tau"], "size": os.path.getsize("models/tfidf_lr.joblib") / 1e6,
})

# 2) Feedforward network on averaged GloVe (Area A)
kv = load_glove()
ck = torch.load("models/ffn.pt", map_location="cpu")
ffn_model = make_model()
ffn_model.load_state_dict(ck["state_dict"])
ffn_model.eval()
f_mean, f_std = ck["mean"].numpy(), ck["std"].numpy()


def ffn_fn(bt):
    X = ((featurize(bt, kv) - f_mean) / f_std).astype(np.float32)
    with torch.inference_mode():
        logits = ffn_model(torch.from_numpy(X))
    return torch.softmax(logits, dim=-1).numpy()


models.append({
    "name": "FFN (GloVe avg)", "area": "A", "fn": ffn_fn,
    "tau": ck["tau"], "size": os.path.getsize("models/ffn.pt") / 1e6,
})

# 3) BERT-base teacher (Area C)
tok_t = AutoTokenizer.from_pretrained(TEACHER)
bert = AutoModelForSequenceClassification.from_pretrained(TEACHER, dtype=torch.float32).eval()
models.append({
    "name": "BERT-base (fp32)", "area": "C", "fn": hf_batch_fn(bert, tok_t),
    "tau": json.load(open(hf_hub_download(TEACHER, "tau.json")))["tau"],
    "size": size_mb_module(bert),
})

# 4) + 5) DistilBERT student, fp32 and int8 (Area D)
taus = json.load(open("models/student_taus.json"))
tok_s = AutoTokenizer.from_pretrained(STUDENT)
stu = AutoModelForSequenceClassification.from_pretrained(STUDENT, dtype=torch.float32).eval()
stu_q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(stu), {nn.Linear}, dtype=torch.qint8).eval()
models.append({
    "name": "DistilBERT student (fp32)", "area": "D", "fn": hf_batch_fn(stu, tok_s),
    "tau": taus["tau_fp32"], "size": size_mb_module(stu),
})
models.append({
    "name": "DistilBERT student (int8)", "area": "D", "fn": hf_batch_fn(stu_q, tok_s),
    "tau": taus["tau_int8"], "size": size_mb_module(stu_q),
})

rows = []
m_in = y != OOS
for m in models:
    t0 = time.time()
    probs = m["fn"](texts)
    pred = final_predictions(probs, OOS, m["tau"])
    acc = in_scope_accuracy(y, pred, OOS)
    rec = oos_recall(y, pred, OOS)
    am = in_scope_accuracy(y, probs.argmax(axis=1), OOS)
    team_acc = float((TEAM[pred][m_in] == TEAM[y][m_in]).mean())
    lat = latency(m["fn"])
    rows.append({
        "model": m["name"], "area": m["area"], "tau": round(m["tau"], 2),
        "in_scope_acc": round(acc, 4), "oos_recall": round(rec, 4),
        "score": round((acc + rec) / 2, 4), "argmax_acc_no_tau": round(am, 4),
        "team_acc": round(team_acc, 4), "size_mb": round(m["size"], 1),
        "lat_mean_ms": round(lat[0], 1), "lat_median_ms": round(lat[1], 1),
        "lat_p95_ms": round(lat[2], 1),
    })
    print(f"done: {m['name']} [{time.time() - t0:.0f}s]")

res = pd.DataFrame(rows)
res.to_csv("results/results.csv", index=False)
print("\n" + res.to_string(index=False))

# ELIZA rules on the test set
pred_st = te.text.map(match_small_talk)
is_st = te.label_name.map(team_of) == SMALL_TALK_TEAM
ok = int((pred_st[is_st] == te.label_name[is_st]).sum())
wrong = int((pred_st[is_st].notna() & (pred_st[is_st] != te.label_name[is_st])).sum())
miss = int(pred_st[is_st].isna().sum())
false_trig = int(pred_st[~is_st].notna().sum())
eliza = {"small_talk_rows": int(is_st.sum()), "correct": ok, "wrong_intent": wrong,
         "no_match": miss, "other_rows": int((~is_st).sum()), "false_triggers": false_trig}
json.dump(eliza, open("results/eliza_test.json", "w"), indent=1)
print("\nELIZA on test:", eliza)
print("\nsaved results/results.csv, results/eliza_test.json, results/hardware.txt")