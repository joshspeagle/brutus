# Sourced by the A/B scripts. Moves to the repository root, checks that src/
# differs from the baseline ref ($BASE_REF, default HEAD), and checks the ref
# out to a temporary worktree at $BASE_SRC/.. that is removed on exit.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_REF=${BASE_REF:-HEAD}
if git diff --quiet "$BASE_REF" -- src &&
  [ -z "$(git ls-files --others --exclude-standard src)" ]; then
  echo "error: src/ has no changes against $BASE_REF" >&2
  exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"; git worktree prune' EXIT
git worktree add --quiet --detach "$tmp" "$BASE_REF"
BASE_SRC="$tmp/src"
