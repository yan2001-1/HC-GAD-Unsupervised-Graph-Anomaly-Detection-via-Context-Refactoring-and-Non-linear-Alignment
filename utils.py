import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
)

def get_anomaly_scores(x_true, adj_true, x_hat, adj_hat, alpha_score):
    """计算综合异常分数"""
    attr_err = torch.mean((x_true - x_hat)**2, dim=1)
    struct_err = torch.mean((adj_true - adj_hat)**2, dim=1)
    
    # 极大极小归一化以平衡量级
    attr_err = (attr_err - attr_err.min()) / (attr_err.max() - attr_err.min() + 1e-8)
    struct_err = (struct_err - struct_err.min()) / (struct_err.max() - struct_err.min() + 1e-8)
    
    scores = alpha_score * attr_err + (1 - alpha_score) * struct_err
    return scores.cpu().numpy()

def select_threshold(scores, min_ratio=0.005, max_ratio=0.2):
    """UMGAD 二阶差分自动寻找拐点 (Label-free, 带稳健约束)"""
    s = np.asarray(scores).astype(np.float64)
    n = len(s)
    s_sorted = np.sort(s)[::-1]

    w = max(int(np.floor(0.005 * n)), 5)
    cumsum = np.cumsum(np.insert(s_sorted, 0, 0.0))
    s_bar = (cumsum[w:] - cumsum[:-w]) / w   

    d1 = s_bar[:-1] - s_bar[1:]
    d2 = d1[:-1] - d1[1:]

    min_k = max(1, int(np.floor(min_ratio * n)))
    max_k = max(min_k + 1, int(np.floor(max_ratio * n)))

    low = max(min_k - 1, 0)
    high = min(max_k - 1, len(d2) - 1)

    if high >= low:
        local = np.abs(d2[low:high + 1])
        best_idx = low + int(np.argmax(local))
    else:
        mags = np.abs(d2)
        best_idx = int(np.argmax(mags))

    threshold = float(s_bar[best_idx])
    pred_labels = (s >= threshold).astype(int)

    # 若出现极端分割，回退到受限 top-k 切分
    k_pred = int(pred_labels.sum())
    if k_pred < min_k or k_pred > max_k:
        k = min(max(k_pred, min_k), max_k)
        threshold = float(s_sorted[k - 1])
        pred_labels = np.zeros_like(s, dtype=int)
        pred_labels[np.argsort(s)[-k:]] = 1

    return threshold, pred_labels

def evaluate(labels, scores):
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

def evaluate_binary(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    return acc, precision, recall, f1
