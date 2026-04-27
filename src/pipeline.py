from .preprocessing import prepare_dataframe
from .embedding import embed_masked, embed_cls, embed_tfidf
from .clustering import (
    find_best_k,
    cluster_embeddings,
    find_best_k_agglomerative,
    cluster_agglomerative,
    find_best_k_hybrid
)
from .metrics import compute_metrics
import numpy as np


def run_pipeline(
    df_anno,
    tokenizer,
    model,
    device,
    method="kmeans",
    k_mode="auto",
    forced_k=None,
    embedding_type="mask"
):
    # -------------------------
    # 1. PREPROCESSING
    # -------------------------
    texts = df_anno["context"].tolist()
    term = df_anno["word"].iloc[0]

    df_term = prepare_dataframe(texts, term)
    anon_texts = df_term["anon_text"].tolist()

    labels_gold = df_anno.loc[df_term["index"], "gold_sense"].tolist()

    # -------------------------
    # 2. EMBEDDINGS
    # -------------------------
    if embedding_type == "mask":
        X = embed_masked(anon_texts, tokenizer, model, device)

    elif embedding_type == "cls":
        X = embed_cls(anon_texts, tokenizer, model, device)

    elif embedding_type == "tfidf":
        X = embed_tfidf(anon_texts)

    else:
        raise ValueError("embedding_type must be 'mask' or 'cls'")

    # -------------------------
    # 3. SELECT K
    # -------------------------
    if k_mode == "auto":

        if method == "kmeans":
            best_k, _, _ = find_best_k(X)

        elif method == "agglo":
            best_k, _, _ = find_best_k_agglomerative(X)

        elif method == "hybrid":
            best_k, _, _ = find_best_k_hybrid(X)

        else:
            best_k = 2

        setting = "auto_k"

    elif k_mode == "forced":
        best_k = forced_k
        setting = "forced_k"

    # -------------------------
    # 4. CLUSTERING
    # -------------------------
    if method == "kmeans":
        labels_pred, sil = cluster_embeddings(X, best_k)

    elif method == "agglo":
        labels_pred, sil = cluster_agglomerative(X, best_k)

    elif method == "hybrid":
        labels_pred, sil = cluster_embeddings(X, best_k)

    elif method == "baseline":
        best_k = 1
        labels_pred = np.zeros(len(X), dtype=int)
        sil = 0.0
        setting = "baseline"

    elif method == "random":
        rng = np.random.RandomState(42)
        k = len(np.unique(labels_gold))  # или 2-5
        labels_pred = rng.randint(0, k, size=len(X))
        best_k = 3
        setting = "random_baseline"

    else:
        raise ValueError(f"Unknown method: {method}")

    # -------------------------
    # 5. METRICS
    # -------------------------
    metrics = compute_metrics(
        labels_gold,
        labels_pred,
        X,
        term,
        best_k,
        setting
    )

    return {
        "metrics": metrics,
        "labels": labels_pred,
        "texts": anon_texts,
        "gold": labels_gold,
        "X": X
    }