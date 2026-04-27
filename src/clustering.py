import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

def find_best_k(X, k_min=1, k_max=5):
    best_k = k_min
    best_score = -1
    scores = {}

    max_k = min(k_max, len(X) - 1)

    for k in range(k_min, max_k + 1):
        if k == 1:
            score = 0.0
        else:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            score = silhouette_score(X, labels)

        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    if best_score < 0.1:
        return 1, 0.0, scores

    return best_k, best_score, scores


def cluster_embeddings(X, n_clusters):
    n_clusters = min(n_clusters, len(X) - 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
    return labels, sil



def cluster_agglomerative(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
    return labels, sil

def find_best_k_agglomerative(X, k_min=2, k_max=5):
    best_k = k_min
    best_score = -1
    scores = {}

    max_k = min(k_max, len(X) - 1)

    for k in range(k_min, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        score = silhouette_score(X, labels)

        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores


def find_best_k_hybrid(X, k_min=2, k_max=6):
    best_k = k_min
    best_score = -1
    scores = {}

    for k in range(k_min, min(k_max, len(X)-1)+1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        score = silhouette_score(X, labels)

        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores