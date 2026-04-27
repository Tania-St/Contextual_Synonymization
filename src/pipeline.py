
from .preprocessing import prepare_dataframe
from .embedding import embed_masked
from .clustering import (
    find_best_k,
    cluster_embeddings,
    find_best_k_agglomerative,
    cluster_agglomerative
)
from .metrics import compute_metrics

def run_pipeline(
    df_anno,
    tokenizer,
    model,
    device,
    method="kmeans",
    k_mode="auto",
    forced_k=None
):
    # --- 1. данные ---
    texts = df_anno["context"].tolist()
    term = df_anno["word"].iloc[0]

    df_term = prepare_dataframe(texts, term)
    anon_texts = df_term["anon_text"].tolist()

    # синхронизация gold labels
    labels_gold = df_anno.loc[df_term["index"], "gold_sense"].tolist()

    # --- 2. embeddings ---
    X = embed_masked(anon_texts, tokenizer, model, device)

    # --- 3. выбор k ---
    if k_mode == "auto":

        if method == "kmeans":
            best_k, _, _ = find_best_k(X)

        elif method == "agglo":
            best_k, _, _ = find_best_k_agglomerative(X)

        setting = "auto_k"

    elif k_mode == "forced":
        best_k = forced_k
        setting = "forced_k"

    # --- 4. кластеризация ---
    if method == "kmeans":
        labels_pred, sil = cluster_embeddings(X, best_k)

    elif method == "agglo":
        labels_pred, sil = cluster_agglomerative(X, best_k)

    else:
        raise ValueError(f"Unknown method: {method}")

    # --- 5. метрики ---
    metrics = compute_metrics(
        labels_gold,
        labels_pred,
        X,
        term,
        best_k,
        setting
    )
    return metrics
    # --- 6. return ---
    return {
        "metrics": metrics,
        #"labels_pred": labels_pred,
        #"labels_gold": labels_gold,
        #"X": X,
        #"texts": anon_texts
    }