import re
from collections import Counter

import numpy as np
import pandas as pd

from preprocess import normalize

GLOVE_NAME = "glove-wiki-gigaword-100"
TOKEN_RE = re.compile(r"n't|[a-z0-9]+|'[a-z]+")


def load_glove():
    import gensim.downloader as api
    return api.load(GLOVE_NAME)   # ilk seferde indirir, sonra cache'ten açar


def tokenize(text):
    t = re.sub(r"n't\b", " n't", normalize(text))   # can't -> ca n't (GloVe formatı)
    return TOKEN_RE.findall(t)


def sent_vec(text, kv):
    vecs = [kv[t] for t in tokenize(text) if t in kv.key_to_index]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def featurize(texts, kv):
    return np.stack([sent_vec(t, kv) for t in texts])


if __name__ == "__main__":
    kv = load_glove()
    print("vocab size:", len(kv.key_to_index), "| vector size:", kv.vector_size)
    print("most similar to 'dishwasher':", [w for w, _ in kv.most_similar("dishwasher", topn=5)])

    for split in ["train", "validation", "test"]:
        df = pd.read_csv(f"data/raw/{split}.csv")
        X = featurize(df.text, kv)
        np.save(f"data/raw/glove100_{split}.npy", X)
        zero_rows = int((np.abs(X).sum(axis=1) == 0).sum())
        print(f"{split}: features {X.shape} | rows with no known word: {zero_rows}")

    tr = pd.read_csv("data/raw/train.csv")
    counts = Counter(t for x in tr.text for t in tokenize(x))
    oov = {t: c for t, c in counts.items() if t not in kv.key_to_index}
    cover = 1 - sum(oov.values()) / sum(counts.values())
    print(f"train token coverage: {cover:.4f}")
    print("most frequent unknown tokens:", sorted(oov.items(), key=lambda x: -x[1])[:15])