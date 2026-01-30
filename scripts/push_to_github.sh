#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/push_to_github.sh <github_repo_url> [branch]"
  echo "Example: ./scripts/push_to_github.sh https://github.com/<user>/<repo>.git main"
  exit 1
fi

REPO_URL="$1"
BRANCH="${2:-main}"

# Initialize git repo if needed
if [[ ! -d .git ]]; then
  git init
fi

git add .
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Add RAP-Lite trainable baseline"
fi

git branch -M "$BRANCH" || true

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git push -u origin "$BRANCH"
echo "✅ Pushed to $REPO_URL ($BRANCH)"
