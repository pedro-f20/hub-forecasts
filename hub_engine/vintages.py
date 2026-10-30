"""Real data vintages from a hub's git history: every committed version of a truth file.

The commit time is when the version became public, so `as_of(t)` = newest commit strictly before t.
Usage: python -m hub_engine.vintages <owner/repo> <path-in-repo> <since YYYY-MM-DD> <out_dir>
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd


def list_commits(repo: str, path: str, since: str) -> list[tuple[str, str]]:
    out = subprocess.run(["gh", "api", "--paginate", f"repos/{repo}/commits?path={path}&since={since}T00:00:00Z&per_page=100",
                          "--jq", ".[] | [.sha, .commit.committer.date] | @tsv"], capture_output=True, text=True, check=True)
    return [tuple(line.split("\t")) for line in out.stdout.splitlines() if line]


def fetch(repo: str, path: str, since: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for sha, date in list_commits(repo, path, since):
        dest = out_dir / f"{date.replace(':', '')}_{sha[:8]}.csv"
        if not dest.exists():
            try:
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}") as r:
                    dest.write_bytes(r.read())
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                print(f"[vintages] {sha[:8]} {date}: file removed in this commit, skipped")
                continue
        index.append({"committed": date, "sha": sha, "file": dest.name})
    (out_dir / "index.json").write_text(json.dumps(sorted(index, key=lambda x: x["committed"]), indent=1))
    print(f"[vintages] {repo}:{path}: {len(index)} versions -> {out_dir}")


def as_of(vintage_dir: Path, cutoff: pd.Timestamp) -> Path | None:
    """Newest version committed strictly before the cutoff (UTC), or None."""
    index = json.loads((vintage_dir / "index.json").read_text())
    seen = [v for v in index if pd.Timestamp(v["committed"]).tz_convert(None) < cutoff]
    return vintage_dir / seen[-1]["file"] if seen else None


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]))
