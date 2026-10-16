import numpy as np
import pytest

from budget_sid.semantic_ids import append_tiger_dedup_digit, rq_kmeans, semantic_ids_from_embeddings


def test_rq_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(4)[:3] * 10
    points = np.vstack([center + rng.normal(scale=0.01, size=(12, 4)) for center in centers])
    codes, codebooks = rq_kmeans(points, n_codebooks=1, codebook_width=3, n_iter=30, seed=0)
    assert codes.shape == (36, 1)
    assert codebooks[0].shape == (3, 4)
    for start in (0, 12, 24):
        assert len(set(codes[start : start + 12, 0].tolist())) == 1
    assert len(set(codes[:, 0].tolist())) == 3


def test_tiger_dedup_digit_resolves_collisions():
    codes = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
        ]
    )
    semantic_ids = append_tiger_dedup_digit(codes, codebook_width=256)
    assert semantic_ids.shape == (4, 4)
    assert semantic_ids[2, 3] == 0
    assert semantic_ids[0, 3] == 0
    assert semantic_ids[1, 3] == 1
    assert semantic_ids[3, 3] == 2
    assert len({tuple(row) for row in semantic_ids.tolist()}) == 4


def test_duplicate_embeddings_receive_distinct_ids():
    embeddings = np.random.default_rng(1).normal(size=(6, 8))
    embeddings[3] = embeddings[0]
    embeddings[4] = embeddings[0]
    semantic_ids = semantic_ids_from_embeddings(
        embeddings, n_codebooks=3, codebook_width=8, n_iter=8, seed=1
    )
    assert semantic_ids.shape == (6, 4)
    assert np.array_equal(semantic_ids[0, :3], semantic_ids[3, :3])
    assert np.array_equal(semantic_ids[0, :3], semantic_ids[4, :3])
    assert len({int(semantic_ids[0, 3]), int(semantic_ids[3, 3]), int(semantic_ids[4, 3])}) == 3


def test_dedup_digit_rejects_oversized_collision_group():
    codes = np.tile(np.array([[1, 1, 1]]), (5, 1))
    with pytest.raises(ValueError, match="collision group"):
        append_tiger_dedup_digit(codes, codebook_width=4)
