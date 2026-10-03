"""Exercise the frozen sidecar's update verification mode without starting a host."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("commit_sha")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit_sha):
        parser.error("Expected exact source commit")
    with tempfile.TemporaryDirectory(prefix="jarvis-binary-probe-") as directory:
        output = Path(directory).resolve() / "result.json"
        subprocess.run([str(args.binary.resolve()), "--update-self-test", str(output)],
                       check=True, timeout=120, shell=False)
        result = json.loads(output.read_text("utf-8"))
        version = json.loads(Path("web/src-tauri/tauri.conf.json").read_text("utf-8"))["version"]
        if result != {"ok": True, "version": version, "commit_sha": args.commit_sha}:
            raise RuntimeError("Packaged sidecar did not verify its exact source build")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
