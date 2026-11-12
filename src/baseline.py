import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from metrics import tune_tau
from preprocess import normalize

names = json.load(open("data/raw/label_names.json"))
OOS = names.index("oos")

tr = pd.read_csv("data/raw/train.csv")
va = pd.read_csv("data/raw/validation.csv")


def run(use_norm, C):
    f = normalize if use_norm else (lambda s: s)
    t0 = time.time()
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000),
    )
    pipe.fit(tr.text.map(f), tr.intent)
    assert list(pipe.classes_) == list(range(len(names)))
    probs = pipe.predict_proba(va.text.map(f))
    score, tau, acc, rec = tune_tau(probs, va.intent.values, OOS)
    print(
        f"normalize={str(use_norm):5} C={C:<4} tau={tau:.2f} "
        f"in-scope acc={acc:.4f} oos recall={rec:.4f} score={score:.4f} "
        f"[{time.time() - t0:.0f}s]"
    )
    return score, tau, pipe


best = None
for use_norm in [False, True]:
    for C in [10, 100]:
        score, tau, pipe = run(use_norm, C)
        if use_norm and (best is None or score > best["score"]):
            best = {"score": score, "tau": tau, "C": C, "pipe": pipe}

joblib.dump(best, "models/tfidf_lr.joblib")
print("saved models/tfidf_lr.joblib | C =", best["C"], "| tau =", round(best["tau"], 2))