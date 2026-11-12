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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from metrics import in_scope_accuracy, tune_tau
from preprocess import normalize

STUDENT = "goncayldz/clinc150-distilbert-student"
MAX_LEN = 40

if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"

names = json.load(open("data/raw/label_names.json"))
OOS = names.index("oos")
va = pd.read_csv("data/raw/validation.csv")
y = va.intent.values
texts = [normalize(t) for t in va.text]

tok = AutoTokenizer.from_pretrained(STUDENT)
fp32 = AutoModelForSequenceClassification.from_pretrained(STUDENT, dtype=torch.float32).eval()
int8 = torch.ao.quantization.quantize_dynamic(
    copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8
).eval()


def size_mb(m):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(m.state_dict(), path)
    s = os.path.getsize(path) / 1e6
    os.remove(path)
    return s


@torch.inference_mode()
def probs_of(m, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        logits = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        out.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(out)


@torch.inference_mode()
def run_one(m, text):
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits


def latency(m, n=200, warm=20):
    for i in range(warm):
        run_one(m, texts[i % 20])
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        run_one(m, texts[i % 20])
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), statistics.median(ts), float(np.percentile(ts, 95))


try:
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
except Exception:
    cpu = platform.processor()
print("hardware:", cpu, "|", platform.platform(), "| torch threads:", torch.get_num_threads())
print("engine:", torch.backends.quantized.engine)

res = {}
for tag, m in [("fp32", fp32), ("int8", int8)]:
    t0 = time.time()
    p = probs_of(m)
    score, tau, acc, rec = tune_tau(p, y, OOS)
    am = in_scope_accuracy(y, p.argmax(axis=1), OOS)
    lat = latency(m)
    res[tag] = {
        "size_mb": size_mb(m), "argmax_acc": am, "score": score, "tau": tau,
        "in_scope_acc": acc, "oos_recall": rec,
        "lat_mean": lat[0], "lat_median": lat[1], "lat_p95": lat[2],
        "pred": p.argmax(axis=1),
    }
    print(f"\n[{tag}] size {res[tag]['size_mb']:.1f} MB | val argmax acc {am:.4f} | "
          f"score {score:.4f} (acc {acc:.4f}, oos recall {rec:.4f}, tau {tau:.2f})")
    print(f"[{tag}] latency ms per message: mean {lat[0]:.1f} | median {lat[1]:.1f} | p95 {lat[2]:.1f} "
          f"[{time.time() - t0:.0f}s]")

agree = float((res["fp32"]["pred"] == res["int8"]["pred"]).mean())
print(f"\nfp32 vs int8 same prediction on validation: {agree:.4f}")
print(f"size ratio fp32/int8: {res['fp32']['size_mb'] / res['int8']['size_mb']:.2f}x | "
      f"speed ratio (mean latency) fp32/int8: {res['fp32']['lat_mean'] / res['int8']['lat_mean']:.2f}x")

tau_hub = json.load(open(hf_hub_download(STUDENT, "tau.json")))["tau"]
print("tau stored on Hub for fp32 student:", tau_hub)

os.makedirs("models", exist_ok=True)
json.dump({"tau_int8": res["int8"]["tau"], "tau_fp32": res["fp32"]["tau"]},
          open("models/student_taus.json", "w"))
print("saved models/student_taus.json")