import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

def purity_score(y_true, y_pred):
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    total = 0

    for cluster in df["pred"].unique():
        subset = df[df["pred"] == cluster]
        most_common = subset["true"].value_counts().max()
        total += most_common

    return total / len(df)


def pairwise_metrics(labels_gold, labels_pred):
    pairs = []

    for i1, i2 in combinations(range(len(labels_pred)), 2):
        gold_same = int(labels_gold[i1] == labels_gold[i2])
        pred_same = int(labels_pred[i1] == labels_pred[i2])
        pairs.append((gold_same, pred_same))

    df_pairs = pd.DataFrame(pairs, columns=["gold_same", "pred_same"])

    precision = precision_score(df_pairs["gold_same"], df_pairs["pred_same"])
    recall = recall_score(df_pairs["gold_same"], df_pairs["pred_same"])
    f1 = f1_score(df_pairs["gold_same"], df_pairs["pred_same"])

    return precision, recall, f1


def compute_metrics(labels_gold, labels_pred, X, term, k, setting):
    precision, recall, f1 = pairwise_metrics(labels_gold, labels_pred)

    ari = adjusted_rand_score(labels_gold, labels_pred)
    nmi = normalized_mutual_info_score(labels_gold, labels_pred)
    purity = purity_score(labels_gold, labels_pred)

    sil = silhouette_score(X, labels_pred) if len(np.unique(labels_pred)) > 1 else 0

    return {
        "term": term,
        "setting": setting,
        "k": k,
        "silhouette": sil,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "ARI": ari,
        "NMI": nmi,
        "purity": purity
    }