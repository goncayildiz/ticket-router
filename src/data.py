import json
from pathlib import Path
from datasets import load_dataset

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

try:
    ds = load_dataset("clinc/clinc_oos", "plus")
except Exception as e:
    print("clinc/clinc_oos failed:", e)
    ds = load_dataset("clinc_oos", "plus")

names = ds["train"].features["intent"].names
oos_id = names.index("oos")

for split in ["train", "validation", "test"]:
    df = ds[split].to_pandas()
    df["label_name"] = df["intent"].map(lambda i: names[i])
    df.to_csv(RAW / f"{split}.csv", index=False)
    n_oos = int((df["intent"] == oos_id).sum())
    print(f"{split}: {len(df)} rows | oos: {n_oos} | in-scope: {len(df) - n_oos}")

json.dump(names, open(RAW / "label_names.json", "w"), indent=0)
print("number of labels:", len(names), "| oos id:", oos_id)
print(df.head())