#!/usr/bin/env bash
# Optional. Push the current branch to the public Budget-SID repository.
# Public source of truth: https://github.com/JOrangsepha/Budget-SID
# Does not force-push and refuses to publish a vendored GRID tree.

set -euo pipefail

PUBLIC_URL="https://github.com/JOrangsepha/Budget-SID.git"
REMOTE_NAME="${1:-origin}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  git remote add "${REMOTE_NAME}" "${PUBLIC_URL}"
fi

echo "Public remote (${REMOTE_NAME}): $(git remote get-url "${REMOTE_NAME}")"
echo "Expected public source of truth: ${PUBLIC_URL}"

if git diff --cached --name-only | grep -q '^third_party/GRID/'; then
  echo "Refusing to push vendored GRID sources." >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push -u "${REMOTE_NAME}" "${BRANCH}"
