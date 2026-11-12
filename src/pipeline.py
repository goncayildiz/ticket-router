import warnings

warnings.filterwarnings("ignore")

import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from embed import featurize, load_glove
from preprocess import PRODUCTS, normalize, preprocess
from smalltalk import match_small_talk, small_talk_reply
from teams import HUMAN_TEAM, SMALL_TALK_TEAM, team_of

ROOT = Path(__file__).resolve().parent.parent
TEACHER = "goncayldz/clinc150-bert-teacher"
STUDENT = "goncayldz/clinc150-distilbert-student"
CLASSICAL = "goncayldz/clinc150-classical-models"
MAX_LEN = 40


def make_ffn(d_in=100, hidden=(512, 256), p=0.3, n_out=151):
    # src/ffn.py ile ayni mimari (state_dict anahtarlari eslesmeli)
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def known_words_from_hub(min_count=2):
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    counts = Counter(w for t in ds["text"] for w in normalize(t).split())
    return {w for w, c in counts.items() if c >= min_count} | set(PRODUCTS)


class Router:
    def __init__(self):
        for e in ("fbgemm", "x86", "qnnpack"):
            if e in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = e
                break

        csv = ROOT / "results" / "results.csv"
        taus = pd.read_csv(csv).set_index("model")["tau"]

        def tau(key):
            return float(taus[key])

        print("[1/6] DistilBERT student -> int8 (dynamic quantization)")
        tok_s = AutoTokenizer.from_pretrained(STUDENT)
        stu = AutoModelForSequenceClassification.from_pretrained(STUDENT, dtype=torch.float32).eval()
        id2label = {int(k): v for k, v in stu.config.id2label.items()}
        self.names = [id2label[i] for i in range(len(id2label))]
        self.oos = self.names.index("oos")
        stu_q = torch.ao.quantization.quantize_dynamic(stu, {nn.Linear}, dtype=torch.qint8).eval()
        del stu

        print("[2/6] BERT-base teacher")
        tok_t = AutoTokenizer.from_pretrained(TEACHER)
        bert = AutoModelForSequenceClassification.from_pretrained(TEACHER, dtype=torch.float32).eval()

        print("[3/6] tf-idf + LogReg and FFN from the Hub")
        lr = joblib.load(hf_hub_download(CLASSICAL, "tfidf_lr.joblib"))["pipe"]
        ck = torch.load(hf_hub_download(CLASSICAL, "ffn.pt"), map_location="cpu")
        ffn = make_ffn()
        ffn.load_state_dict(ck["state_dict"])
        ffn.eval()
        f_mean, f_std = ck["mean"].numpy(), ck["std"].numpy()

        print("[4/6] GloVe vectors")
        kv = load_glove()

        print("[5/6] known-word list for spelling correction")
        self.known = known_words_from_hub()

        def run_hf(model, tok, text):
            enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.inference_mode():
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            return torch.softmax(logits.float(), dim=-1)[0].numpy()

        def run_lr(text):
            return lr.predict_proba([text])[0]

        def run_ffn(text):
            X = ((featurize([text], kv) - f_mean) / f_std).astype(np.float32)
            with torch.inference_mode():
                logits = ffn(torch.from_numpy(X))
            return torch.softmax(logits, dim=-1)[0].numpy()

        # (gorunen ad, fonksiyon, tau) - tau'lar validation'da secildi, results.csv'den okunur
        self.models = [
            ("tf-idf + LogReg (B, fp32)", run_lr, tau("tfidf + LogReg")),
            ("FFN on GloVe (A, fp32)", run_ffn, tau("FFN (GloVe avg)")),
            ("BERT-base (C, fp32)", lambda t: run_hf(bert, tok_t, t), tau("BERT-base (fp32)")),
            ("DistilBERT student (D, int8)", lambda t: run_hf(stu_q, tok_s, t),
             tau("DistilBERT student (int8)")),
        ]

        print("[6/6] warm-up")
        self.analyze("hello there")
        self.analyze("my dishwasher is not draining")
        print("ready")

    def analyze(self, message):
        message = (message or "").strip()
        prep = preprocess(message, self.known)
        clean = prep["clean"]
        out = {"prep": prep, "rows": [], "top3": [], "decision": None}

        if not clean:
            out["decision"] = {"team": HUMAN_TEAM, "source": "empty message after cleaning", "reply": None}
            return out

        for name, fn, tau in self.models:
            t0 = time.perf_counter()
            p = fn(clean)
            ms = (time.perf_counter() - t0) * 1000
            i = int(p.argmax())
            conf = float(p[i])
            intent = self.names[i]
            low = (i == self.oos) or (conf < tau)
            team = HUMAN_TEAM if low else team_of(intent)
            out["rows"].append({"model": name, "intent": intent, "conf": conf, "tau": tau,
                                "team": team, "ms": ms, "oos": i == self.oos})
        top = np.argsort(-p)[:3]          # p = son model (int8 student) cikisi
        out["top3"] = [(self.names[int(j)], float(p[int(j)])) for j in top]

        final = out["rows"][-1]            # R5: karari quantized model verir
        rule_intent = match_small_talk(message)
        if rule_intent:
            out["decision"] = {"team": SMALL_TALK_TEAM, "source": "ELIZA rule: " + rule_intent,
                               "reply": small_talk_reply(rule_intent)}
        elif final["team"] == SMALL_TALK_TEAM:
            out["decision"] = {"team": SMALL_TALK_TEAM, "source": "int8 student: " + final["intent"],
                               "reply": small_talk_reply(final["intent"])}
        elif final["team"] == HUMAN_TEAM:
            why = "out-of-scope" if final["oos"] else f"low confidence ({final['conf']:.2f} < {final['tau']:.2f})"
            out["decision"] = {"team": HUMAN_TEAM, "source": "int8 student: " + why, "reply": None}
        else:
            out["decision"] = {"team": final["team"],
                               "source": f"int8 student: {final['intent']} ({final['conf']:.2f})",
                               "reply": None}
        return out