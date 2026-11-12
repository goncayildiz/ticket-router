import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metrics import in_scope_accuracy, tune_tau

EPOCHS = 60
BATCH = 64
LR = 0.01
HIDDEN = (512, 256)
DROPOUT = 0.3

names = json.load(open("data/raw/label_names.json"))
OOS = names.index("oos")
N_CLASSES = len(names)


def make_model(d_in=100, hidden=HIDDEN, p=DROPOUT, n_out=N_CLASSES):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
    return torch.softmax(logits, dim=1).numpy()


def load_split(split):
    X = np.load(f"data/raw/glove100_{split}.npy")
    y = pd.read_csv(f"data/raw/{split}.csv").intent.values
    return X, y


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    Xtr, ytr = load_split("train")
    Xva, yva = load_split("validation")
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-6
    Xtr_s = ((Xtr - mean) / std).astype(np.float32)
    Xva_s = ((Xva - mean) / std).astype(np.float32)

    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr).long()),
        batch_size=BATCH,
        shuffle=True,
    )

    best = {"score": -1.0}
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        sched.step()

        probs = predict_probs(model, Xva_s)
        score, tau, acc, rec = tune_tau(probs, yva, OOS)
        if score > best["score"]:
            best = {
                "score": score, "tau": tau, "acc": acc, "rec": rec, "epoch": epoch,
                "argmax_acc": in_scope_accuracy(yva, probs.argmax(axis=1), OOS),
                "state": {k: v.clone() for k, v in model.state_dict().items()},
            }
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:2d} | train loss {total / len(ytr):.4f} | "
                f"val score {score:.4f} (acc {acc:.4f}, oos recall {rec:.4f}, tau {tau:.2f}) "
                f"[{time.time() - t0:.0f}s]"
            )

    print(
        f"\nBEST epoch {best['epoch']} | tau {best['tau']:.2f} | "
        f"in-scope acc {best['acc']:.4f} | oos recall {best['rec']:.4f} | "
        f"score {best['score']:.4f} | argmax in-scope acc (no tau) {best['argmax_acc']:.4f}"
    )

    torch.save(
        {
            "state_dict": best["state"],
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "tau": best["tau"],
            "hidden": list(HIDDEN),
        },
        "models/ffn.pt",
    )
    print("saved models/ffn.pt")


if __name__ == "__main__":
    main()