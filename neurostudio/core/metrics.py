"""回归与分类评价指标（纯 numpy 实现，避免额外运行时依赖）。"""

from __future__ import annotations

import numpy as np


def _as_float_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size != pred.size:
        raise ValueError("真实值与预测值长度不一致")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    true, pred = _as_float_arrays(y_true, y_pred)
    if true.size == 0:
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "medae": 0.0,
            "explained_variance": 0.0,
            "bias": 0.0,
            "count": 0,
        }
    error = pred - true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    medae = float(np.median(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    variance = float(np.var(true))
    explained_variance = 1.0 - float(np.var(error)) / variance if variance > 1e-12 else 0.0
    denom = np.where(np.abs(true) < 1e-9, np.nan, np.abs(true))
    mape = float(np.nanmean(np.abs(error) / denom) * 100) if np.isfinite(denom).any() else 0.0
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": float(r2),
        "mape": mape,
        "medae": medae,
        "explained_variance": float(explained_variance),
        "bias": float(np.mean(error)),
        "count": int(true.size),
    }


# ---------------------------------------------------------------------------
# 分类辅助指标
# ---------------------------------------------------------------------------


def _binary_auc(y_true_binary, scores) -> float | None:
    """Mann-Whitney U 形式的二分类 AUC，正确处理并列分数。"""
    y = np.asarray(y_true_binary, dtype=bool).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def roc_auc_score(y_true, probabilities, n_classes: int | None = None) -> float | None:
    """二分类 AUC，或多分类 one-vs-rest 宏平均 AUC。"""
    true = np.asarray(y_true).astype(np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.shape[0] != true.size:
        return None
    if n_classes is None:
        n_classes = max(int(true.max()) + 1, probs.shape[1])
    n_classes = int(n_classes)
    if probs.shape[1] < n_classes:
        return None
    aucs: list[float] = []
    for index in range(n_classes):
        mask = true == index
        if mask.sum() == 0 or mask.sum() == true.size:
            continue
        auc = _binary_auc(mask, probs[:, index])
        if auc is not None:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else None


def log_loss_score(y_true, probabilities, n_classes: int | None = None) -> float | None:
    true = np.asarray(y_true).astype(np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.shape[0] != true.size:
        return None
    if n_classes is not None and probs.shape[1] < int(n_classes):
        return None
    if true.size == 0 or true.min() < 0 or true.max() >= probs.shape[1]:
        return None
    picked = probs[np.arange(true.size), true]
    picked = np.clip(picked, 1e-15, 1.0)
    return float(-np.mean(np.log(picked)))


def matthews_corrcoef(y_true, y_pred, n_classes: int) -> float:
    true = np.asarray(y_true).astype(np.int64).reshape(-1)
    pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    if true.size == 0:
        return 0.0
    n_classes = max(int(n_classes), int(true.max()) + 1, int(pred.max()) + 1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(confusion, (true, pred), 1)
    total = confusion.sum()
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    numerator = float(np.trace(confusion) * total - np.sum(row_sums * col_sums))
    denominator = float(
        np.sqrt(
            max(total**2 - np.sum(row_sums**2), 0.0)
            * max(total**2 - np.sum(col_sums**2), 0.0)
        )
    )
    return numerator / denominator if denominator > 1e-15 else 0.0


def cohen_kappa(y_true, y_pred, n_classes: int) -> float:
    true = np.asarray(y_true).astype(np.int64).reshape(-1)
    pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    if true.size == 0:
        return 0.0
    n_classes = max(int(n_classes), int(true.max()) + 1, int(pred.max()) + 1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(confusion, (true, pred), 1)
    total = float(confusion.sum())
    if total <= 0:
        return 0.0
    observed = float(np.trace(confusion) / total)
    expected = float(np.sum(confusion.sum(axis=1) * confusion.sum(axis=0)) / (total * total))
    return (observed - expected) / (1.0 - expected) if 1.0 - expected > 1e-15 else 0.0


def classification_metrics(
    y_true,
    y_pred,
    n_classes: int | None = None,
    class_names: list[str] | None = None,
    probabilities=None,
) -> dict:
    true = np.asarray(y_true).astype(np.int64).reshape(-1)
    pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    if true.size != pred.size:
        raise ValueError("真实值与预测值长度不一致")
    if true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "mcc": 0.0,
            "kappa": 0.0,
            "roc_auc": None,
            "log_loss": None,
            "count": 0,
            "confusion": [],
            "per_class": [],
            "class_names": list(class_names or []),
        }

    if n_classes is None:
        n_classes = int(max(true.max(), pred.max())) + 1
    n_classes = max(int(n_classes), int(true.max()) + 1, int(pred.max()) + 1)
    names = list(class_names or [str(i) for i in range(n_classes)])
    names += [str(i) for i in range(len(names), n_classes)]

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    tn = float(true.size) - tp - fp - fn
    support = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    per_class = [
        {
            "index": i,
            "name": names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "specificity": float(specificity[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    ]
    present = support > 0
    macro = lambda values: float(values[present].mean()) if present.any() else 0.0

    auc = None
    log_loss = None
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])
        if probs.shape[0] == true.size:
            auc = roc_auc_score(true, probs, n_classes)
            log_loss = log_loss_score(true, probs, n_classes)

    return {
        "accuracy": float(tp.sum() / max(true.size, 1)),
        "precision": macro(precision),
        "recall": macro(recall),
        "f1": macro(f1),
        "specificity": macro(specificity),
        "balanced_accuracy": macro(recall),
        "mcc": matthews_corrcoef(true, pred, n_classes),
        "kappa": cohen_kappa(true, pred, n_classes),
        "roc_auc": auc,
        "log_loss": log_loss,
        "count": int(true.size),
        "confusion": confusion.tolist(),
        "per_class": per_class,
        "class_names": names[:n_classes],
    }
