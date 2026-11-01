#!/usr/bin/env bash
# One weekly round for each AI4Casting hub: forecast the coming Saturday and open the submission PR from the fork.
# Safe to run several times: skips a hub once its round is submitted or while its data isn't out.
# Needs GH_TOKEN (pedro-f20 classic token, public_repo scope only) and a venv at .venv.
#   REF=YYYY-MM-DD  override the round (a Saturday)
#   DRY_RUN=1       forecast only, no push or PR
#   FINAL=1         last run before the deadline: missing data is an error, not a wait
set -euo pipefail

MODEL=pedro-Chronos2
EXIT_NO_DATA=3
HUBS=("ai4c-hosp:ai4castinghub/hospitalization-forecast" "ai4c-rvdss:ai4castinghub/rvdss-forecast")

# coming Saturday in Toronto, or today if it is one: rounds and deadlines are Toronto time, and a
# delayed Saturday-evening run lands on Sunday in UTC
REF=${REF:-$(python3 -c "import datetime as d, zoneinfo as z; t=d.datetime.now(z.ZoneInfo('America/Toronto')).date(); print(t + d.timedelta((5 - t.weekday()) % 7))")}
FILE="model-output/$MODEL/$REF-$MODEL.csv"
BRANCH="$MODEL-$REF"
[[ -n "${DRY_RUN:-}" || -n "${GH_TOKEN:-}" ]] || { echo "[submit] GH_TOKEN is empty: set the HUB_TOKEN secret" >&2; exit 1; }
failed=0

for entry in "${HUBS[@]}"; do
  hub=${entry%%:*}; upstream=${entry#*:}; fork="pedro-f20/${upstream#*/}"
  echo "[submit] $hub round $REF"
  if [[ -z "${DRY_RUN:-}" ]]; then
    if gh api "repos/$upstream/contents/$FILE" --silent 2>/dev/null; then echo "[submit] $hub: already in the hub"; continue; fi
    open_pr=$(gh pr list --repo "$upstream" --head "$BRANCH" --state open --json url --jq '.[0].url')
    if [[ -n "$open_pr" ]]; then
      # the hub merges by hand, often a week later, so an open PR is normal; a failed check is not
      checks=$(gh pr checks "$open_pr" --json bucket --jq '[.[].bucket] | unique | join(",")' || true)
      if [[ "$checks" == *fail* || "$checks" == *cancel* ]]; then
        echo "[submit] $hub: PR $open_pr failed the hub's validation" >&2; failed=1
      else
        echo "[submit] $hub: PR $open_pr open (checks: ${checks:-none yet}), waiting for merge"
      fi
      continue
    fi
  fi

  rm -rf "hub-$hub" "out-$hub"
  git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$upstream.git" "hub-$hub"
  git -C "hub-$hub" sparse-checkout set target-data hub-config auxiliary-data/target-data-archive
  set +e
  .venv/bin/python forecast.py "$hub" "hub-$hub" "$REF" "out-$hub"
  status=$?
  set -e
  if [[ $status -eq $EXIT_NO_DATA ]]; then
    if [[ -n "${FINAL:-}" ]]; then echo "[submit] $hub: deadline run and still no data" >&2; failed=1; fi
    continue
  fi
  [[ $status -eq 0 ]] || { failed=1; continue; }
  [[ -z "${DRY_RUN:-}" ]] || { echo "[submit] $hub: dry run, stopping before push"; continue; }

  # Branch from the fork's own main, never synced (syncing needs the workflow scope). Safe here because
  # the hub validates the PR's merge into its own main (pull_request event), not the branch alone.
  rm -rf "fork-$hub"
  git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$fork.git" "fork-$hub"
  git -C "fork-$hub" sparse-checkout set "model-output/$MODEL" model-metadata
  mkdir -p "fork-$hub/model-output/$MODEL" "fork-$hub/model-metadata"
  cp "out-$hub/$REF-$MODEL.csv" "fork-$hub/$FILE"
  git -C "fork-$hub" switch -q -c "$BRANCH"
  git -C "fork-$hub" add "$FILE"
  # Until the hub has merged a PR carrying our metadata, every PR must carry it or validation fails
  if ! gh api "repos/$upstream/contents/model-metadata/$MODEL.yml" --silent 2>/dev/null; then
    cp "model-metadata/$MODEL.yml" "fork-$hub/model-metadata/$MODEL.yml"
    git -C "fork-$hub" add "model-metadata/$MODEL.yml"
  fi
  git -C "fork-$hub" -c user.name=pedro-f20 -c user.email=250035667+pedro-f20@users.noreply.github.com \
    commit -q -m "Add $MODEL forecast for $REF"
  # --force: a branch left by a run that pushed but failed to open the PR would otherwise block every retry
  # one hub failing must not stop the other: report it and move on
  git -C "fork-$hub" push -q --force "https://x-access-token:$GH_TOKEN@github.com/$fork.git" "$BRANCH" ||
    { echo "[submit] $hub: push to $fork failed" >&2; failed=1; continue; }
  url=$(gh pr create --repo "$upstream" --head "pedro-f20:$BRANCH" --title "$MODEL forecast $REF" \
    --body "Weekly zero-shot Chronos-2 forecast for the $REF round.") ||
    { echo "[submit] $hub: opening the PR failed" >&2; failed=1; continue; }
  echo "$hub,$REF,$url" >> submissions.csv
  echo "[submit] $hub: $url"
done
exit $failed
