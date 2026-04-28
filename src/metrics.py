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


def safe_silhouette(X, labels):
    labels = np.array(labels)
    n_labels = len(set(labels))

    if n_labels < 2 or n_labels >= len(X):
        return 0.0

    return silhouette_score(X, labels)

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

    precision = precision_score(df_pairs["gold_same"], df_pairs["pred_same"], zero_division=0)
    recall = recall_score(df_pairs["gold_same"], df_pairs["pred_same"], zero_division=0)
    f1 = f1_score(df_pairs["gold_same"], df_pairs["pred_same"], zero_division=0)

    return precision, recall, f1


def compute_metrics(labels_gold, labels_pred, X, term, k, setting):
    precision, recall, f1 = pairwise_metrics(labels_gold, labels_pred)

    ari = adjusted_rand_score(labels_gold, labels_pred)
    nmi = normalized_mutual_info_score(labels_gold, labels_pred)
    purity = purity_score(labels_gold, labels_pred)

    from .metrics import safe_silhouette
    sil = safe_silhouette(X, labels_pred)

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


def show_clusters(texts, labels, gold_labels, n_examples=5):

    df = pd.DataFrame({
        "text": texts,
        "cluster": labels,
        "gold": gold_labels
    })

    for cluster in sorted(df["cluster"].unique()):
        print(f"\ncluster {cluster} ")

        subset = df[df["cluster"] == cluster]

        print("top examples:")
        for t in subset["text"].head(n_examples):
            print("-", t)

        print("gold distribution:")
        print(subset["gold"].value_counts())


def show_mismatches(texts, gold, pred, n=10):

    df = pd.DataFrame({
        "text": texts,
        "gold": gold,
        "pred": pred
    })

    mismatches = df[df["gold"] != df["pred"]]

    print("\n misclassified examples")

    for i, row in mismatches.head(n).iterrows():
        print("\ntext:", row["text"])
        print("gold:", row["gold"], "pred:", row["pred"])