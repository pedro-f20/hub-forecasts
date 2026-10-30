#!/usr/bin/env bash
# Local weekly runner (launchd): uses the machine's gh login instead of a repo secret.
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")"
LOG="$HOME/Library/Logs/hub-forecasts.log"
{
  echo "=== $(date -u +%FT%TZ)"
  git pull -q --ff-only || echo "[run] git pull failed, using local copy"
  [[ -x .venv/bin/python ]] || { uv venv -q --python 3.12 && uv pip install -q --torch-backend cpu -r requirements.txt; }
  GH_TOKEN=$(gh auth token -u pedro-f20) || { echo "[run] no gh token for pedro-f20"; exit 1; }
  export GH_TOKEN
  # Saturday 21:00 local is the last run before the 23:30 Toronto deadline
  [[ $(date +%u) == 6 && $(date +%H) -ge 21 ]] && export FINAL=1
  bash submit.sh
  status=$?
  echo "[run] exit $status"
  if [[ $status -ne 0 ]]; then
    osascript -e 'display notification "Weekly AI4Casting submission failed: see ~/Library/Logs/hub-forecasts.log" with title "hub-forecasts"'
  fi
} >> "$LOG" 2>&1
