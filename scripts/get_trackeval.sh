#!/usr/bin/env bash
# Clone TrackEval at a pinned commit and make it numpy-2 safe.
# Not pip installed on purpose: the PyPI mirrors are stale and disagree on HOTA.
set -euo pipefail
DEST=${1:?usage: get_trackeval.sh <dest-dir>}
COMMIT=${TRACKEVAL_COMMIT:-12c8791b303e0a0b50f753af204249e622d0281a}
mkdir -p "$DEST"
if [ ! -d "$DEST/te/.git" ]; then
  git clone -q https://github.com/JonathonLuiten/TrackEval.git "$DEST/te"
fi
cd "$DEST/te"
git checkout -q "$COMMIT" 2>/dev/null || echo "note: pinned commit unavailable, using default branch"
# numpy>=2 removed the np.float / np.int / np.bool aliases TrackEval still uses
grep -rl 'np\.float\b\|np\.int\b\|np\.bool\b' trackeval/ 2>/dev/null | while read -r f; do
  sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' "$f"
done
python -c "import sys; sys.path.insert(0,'.'); import trackeval; print('TrackEval import OK')"
