import re
from collections import Counter

import pandas as pd

# 1-4 harf + (opsiyonel tire) + 3-5 rakam, ör. DW-6534, RF2210
MODEL_CODE_RE = re.compile(r"\b[A-Za-z]{1,4}-?\d{3,5}[A-Za-z]?\b")

PRODUCTS = [
    "dishwasher", "refrigerator", "fridge", "freezer", "washer", "dryer",
    "oven", "microwave", "stove", "cooktop", "kettle", "toaster", "blender",
    "vacuum", "airfryer", "cooker",
]


def extract_model_code(text):
    """Mesajdaki model kodunu bulur. (kod, koddan arındırılmış metin) döner."""
    m = MODEL_CODE_RE.search(text)
    if not m:
        return None, text
    code = m.group(0).upper()
    return code, text[: m.start()] + " " + text[m.end():]


def normalize(text):
    """Regex ile temizlik. Eğitimde de app'te de aynı fonksiyon kullanılır."""
    t = text.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")   # kıvrık apostrof
    t = re.sub(r"[^\w' ]+|_", " ", t)                      # noktalama -> boşluk
    t = re.sub(r"([^\W\d_])\1{2,}", r"\1\1", t)            # heeello -> heello (sadece harfler)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def edit_distance(a, b):
    """Minimum edit distance (Levenshtein): ekleme, silme, değiştirme = 1."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # silme
                dp[i][j - 1] + 1,         # ekleme
                dp[i - 1][j - 1] + cost,  # değiştirme
            )
    return dp[m][n]


def build_known_words(path="data/raw/train.csv", min_count=2):
    """Train verisinde en az min_count kez geçen kelimeler = 'doğru' kabul edilir."""
    df = pd.read_csv(path)
    counts = Counter(w for t in df.text for w in normalize(t).split())
    known = {w for w, c in counts.items() if c >= min_count}
    return known | set(PRODUCTS)


def correct_token(tok, known):
    """Bilinmeyen kelime bir ürün adına çok yakınsa onu ürün adına çevirir."""
    if tok in known or len(tok) < 5 or tok.isdigit():
        return tok
    best, best_d = tok, 99
    for p in PRODUCTS:
        d = edit_distance(tok, p)
        if d < best_d:
            best, best_d = p, d
    limit = 1 if len(tok) <= 6 else 2   # kısa kelimede daha sıkı eşik
    return best if best_d <= limit else tok


def preprocess(text, known_words=None, spell=True):
    code, no_code = extract_model_code(text)
    clean = normalize(no_code)
    corrections = []
    if spell and known_words is not None:
        out = []
        for tok in clean.split():
            new = correct_token(tok, known_words)
            if new != tok:
                corrections.append((tok, new))
            out.append(new)
        clean = " ".join(out)
    return {"raw": text, "model_code": code, "clean": clean, "corrections": corrections}


if __name__ == "__main__":
    known = build_known_words()
    print("known words:", len(known))
    print("edit_distance(kitten, sitting) =", edit_distance("kitten", "sitting"))
    tests = [
        "My dishwaser is not draining!!! Model DW-6534",
        "the refrigarator makes a strange noise, model RF2210",
        "heeeello, can you help me???",
        "what's the balance on my account",
        "my washer broke",
        "mikrowave does not heat",
        "how much is 1000 dollars in euros",
    ]
    for t in tests:
        print(preprocess(t, known))
    df = pd.read_csv("data/raw/train.csv")
    changed = int((df.text != df.text.map(normalize)).sum())
    print("train rows changed by normalize:", changed, "of", len(df))