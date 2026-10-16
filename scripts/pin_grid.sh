#!/usr/bin/env bash
# Check out Snap Research GRID at the commit Budget-SID is written against.
# The checkout lives in third_party/GRID and is gitignored. This script does
# not vendor GRID and does not modify, reset, or commit upstream files.

set -euo pipefail

COMMIT="2fe3475b2d369580234093f35d52b1a2f54d0472"
REPO="https://github.com/snap-research/GRID.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/GRID"

mkdir -p "${ROOT}/third_party"

if [[ ! -d "${DEST}/.git" ]]; then
  git clone --filter=blob:none "${REPO}" "${DEST}"
fi

if [[ -n "$(git -C "${DEST}" status --porcelain)" ]]; then
  echo "GRID checkout at ${DEST} has local changes. Remove it and re-run; this script will not rewrite GRID." >&2
  exit 1
fi

git -C "${DEST}" fetch --filter=blob:none origin "${COMMIT}"
git -C "${DEST}" checkout --detach "${COMMIT}"

if [[ -n "$(git -C "${DEST}" status --porcelain)" ]]; then
  echo "GRID checkout is dirty after checkout. Refusing to continue." >&2
  exit 1
fi

ACTUAL="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${ACTUAL}" != "${COMMIT}" ]]; then
  echo "GRID pin mismatch: expected ${COMMIT}, got ${ACTUAL}" >&2
  exit 1
fi

echo "GRID pinned at ${ACTUAL}"
echo "Upstream: ${REPO}"
