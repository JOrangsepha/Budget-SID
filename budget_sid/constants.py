"""Pinned upstream identifiers and the frozen search-budget contract."""

GRID_REPO_URL = "https://github.com/snap-research/GRID"
GRID_COMMIT = "2fe3475b2d369580234093f35d52b1a2f54d0472"
PUBLIC_GITHUB_REPO = "https://github.com/JOrangsepha/Budget-SID"

TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

TOP_K = 20
BEAM_WIDTHS = (20, 40, 60)
SMALL_BEAM = 20
LARGE_BEAM = 60
REFERENCE_BEAM = 40

# Average beam width of REFERENCE_BEAM is matched by sending this fraction of
# requests to LARGE_BEAM and the rest to SMALL_BEAM:
# 60p + 20(1-p) = 40 => p = 0.5.
LARGE_BEAM_FRACTION = (REFERENCE_BEAM - SMALL_BEAM) / (LARGE_BEAM - SMALL_BEAM)
