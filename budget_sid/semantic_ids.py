"""CPU residual K-means and the TIGER collision digit.

This is the smoke-test stand-in for GRID's residual quantizer. The
authoritative trainer remains GRID at the pinned commit; nothing here is a
fork of that code.
"""

from __future__ import annotations

import numpy as np


def _kmeans(points: np.ndarray, n_clusters: int, n_iter: int, rng: np.random.Generator):
    n_rows, dim = points.shape
    k = min(int(n_clusters), n_rows)
    if k < 1:
        raise ValueError("n_clusters must be positive")
    centroids = np.empty((k, dim), dtype=np.float64)
    first = int(rng.integers(0, n_rows))
    centroids[0] = points[first]
    for slot in range(1, k):
        distance = ((points[:, None, :] - centroids[None, :slot, :]) ** 2).sum(axis=2)
        nearest = distance.min(axis=1)
        total = float(nearest.sum())
        if total <= 1e-12:
            centroids[slot] = points[int(rng.integers(0, n_rows))]
            continue
        probs = nearest / total
        centroids[slot] = points[int(rng.choice(n_rows, p=probs))]
    labels = np.zeros(n_rows, dtype=np.int64)
    for _ in range(n_iter):
        distance = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distance.argmin(axis=1).astype(np.int64)
        for slot in range(k):
            members = labels == slot
            if not np.any(members):
                centroids[slot] = points[int(rng.integers(0, n_rows))]
            else:
                centroids[slot] = points[members].mean(axis=0)
    return labels, centroids


def rq_kmeans(
    embeddings: np.ndarray,
    n_codebooks: int = 3,
    codebook_width: int = 256,
    n_iter: int = 15,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Residual quantization. Codes have shape ``[n_items, n_codebooks]``."""
    points = np.asarray(embeddings, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("embeddings must have shape [n_items, dim]")
    if n_codebooks < 1:
        raise ValueError("n_codebooks must be positive")
    residual = points.copy()
    codes = []
    codebooks = []
    for layer in range(n_codebooks):
        rng = np.random.default_rng(seed + layer * 1009)
        labels, centroids = _kmeans(residual, codebook_width, n_iter, rng)
        codes.append(labels)
        codebooks.append(centroids)
        residual = residual - centroids[labels]
    return np.stack(codes, axis=1), codebooks


def append_tiger_dedup_digit(codes: np.ndarray, codebook_width: int = 256) -> np.ndarray:
    """Append the extra digit GRID uses to break residual-code collisions.

    Items that share the same codebook tuple receive 0, 1, 2, ... in item-id
    order. Unique codes receive 0. A group larger than ``codebook_width``
    cannot be stored in the TIGER vocabulary and is rejected.
    """
    base = np.asarray(codes)
    if base.ndim != 2 or base.shape[1] < 1:
        raise ValueError("codes must have shape [n_items, n_codebooks]")
    dedup = np.zeros(base.shape[0], dtype=np.int64)
    grouped: dict[tuple, list[int]] = {}
    for index, row in enumerate(base.tolist()):
        grouped.setdefault(tuple(int(value) for value in row), []).append(index)
    for indexes in grouped.values():
        for digit, index in enumerate(indexes):
            if digit >= codebook_width:
                raise ValueError(
                    "a collision group is larger than the codebook width, so the "
                    "TIGER dedup digit does not fit in the vocabulary"
                )
            dedup[index] = digit
    return np.concatenate([base.astype(np.int64), dedup[:, None]], axis=1)


def semantic_ids_from_embeddings(
    embeddings: np.ndarray,
    n_codebooks: int = 3,
    codebook_width: int = 256,
    n_iter: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """3 RQ-KMeans digits plus the TIGER dedup digit. Shape ``[n_items, 4]``."""
    codes, _codebooks = rq_kmeans(
        embeddings,
        n_codebooks=n_codebooks,
        codebook_width=codebook_width,
        n_iter=n_iter,
        seed=seed,
    )
    return append_tiger_dedup_digit(codes, codebook_width=codebook_width)
