"""Generate packaged provenance from the checked-out Git commit."""
from pathlib import Path
import json
import re
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("A complete source commit is required")
    dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root, text=True).strip()
    if dirty:
        raise RuntimeError("Tracked source changes must be committed before packaging")
    version = json.loads((root / "web/src-tauri/tauri.conf.json").read_text("utf-8"))["version"]
    (root / "build-info.json").write_text(json.dumps({"version": version, "commit_sha": commit}, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
