#!/usr/bin/env bash
# Reliable submodule pull that works around the HuggingFace git server's
# "expected 'acknowledgments'" protocol-v2 negotiation failure.
#
# Usage:
#   ./pull_submodules.sh          # update parent + align both submodules
#   ./pull_submodules.sh --init   # also run after a fresh clone
#
# Instead of `git submodule update` (which trips the protocol error), we read
# the exact commit the parent expects for each submodule and fetch/checkout
# that SHA directly. Fetching a specific SHA has been observed to succeed where
# a plain `git fetch origin` fails against huggingface.co.
set -e
cd "$(dirname "$0")"

if [ "${1:-}" = "--init" ]; then
  git submodule init
fi

# Make sure we have the latest parent refs (best-effort; ignore failure).
git fetch origin || true

for sub in additive-rand-transformer maze-transformer; do
  expected=$(git ls-tree HEAD "$sub" | awk '{print $3}')
  if [ -z "$expected" ]; then
    echo "!! could not determine expected commit for $sub; skipping"
    continue
  fi
  echo "=== $sub expects $expected ==="
  (
    cd "$sub"
    # Fetch the exact commit by SHA (avoids the v2 negotiation hang).
    if git fetch origin "$expected" 2>/dev/null; then
      echo "   fetched $expected via SHA"
    else
      echo "   SHA fetch failed, trying plain fetch (may error on HF)..."
      git fetch origin || true
    fi
    # Prefer checkout of the exact SHA; fall back to the branch tip.
    if git cat-file -e "$expected^{commit}" 2>/dev/null; then
      git checkout "$expected"
    else
      git checkout "${expected:0:40}" 2>/dev/null || git checkout -B "main" "origin/main"
    fi
  )
  echo
done

echo "=== submodule sync complete ==="
git submodule status
