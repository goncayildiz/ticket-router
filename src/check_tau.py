import json

import joblib
import pandas as pd

from metrics import final_predictions, in_scope_accuracy, oos_recall
from preprocess import normalize

names = json.load(open("data/raw/label_names.json"))
OOS = names.index("oos")
va = pd.read_csv("data/raw/validation.csv")
m = joblib.load("models/tfidf_lr.joblib")
probs = m["pipe"].predict_proba(va.text.map(normalize))
y = va.intent.values

print("chosen tau:", round(m["tau"], 2))
for tau in [0.0, 0.1, 0.2, 0.3, 0.4, round(m["tau"], 2), 0.6, 0.7, 0.8]:
    p = final_predictions(probs, OOS, tau)
    print(
        f"tau={tau:.2f}  in-scope acc={in_scope_accuracy(y, p, OOS):.4f}  "
        f"oos recall={oos_recall(y, p, OOS):.4f}"
    )