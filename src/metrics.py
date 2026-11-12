import numpy as np


def final_predictions(probs, oos_id, tau):
    """Güven (max olasılık) tau'dan düşükse tahmini 'oos' (Human agent) yap."""
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return np.where(conf < tau, oos_id, pred)


def in_scope_accuracy(y_true, y_pred, oos_id):
    m = y_true != oos_id
    return float((y_pred[m] == y_true[m]).mean())


def oos_recall(y_true, y_pred, oos_id):
    m = y_true == oos_id
    return float((y_pred[m] == oos_id).mean())


def tune_tau(probs, y_true, oos_id):
    """Validation üzerinde tau seç: in-scope accuracy ile OOS recall ortalamasını maksimize et."""
    best = None
    for tau in np.arange(0.0, 1.0, 0.02):
        p = final_predictions(probs, oos_id, tau)
        acc = in_scope_accuracy(y_true, p, oos_id)
        rec = oos_recall(y_true, p, oos_id)
        score = (acc + rec) / 2
        if best is None or score > best[0]:
            best = (score, float(tau), acc, rec)
    return best