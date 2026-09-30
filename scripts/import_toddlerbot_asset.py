"""Import the pinned ToddlerBot 2XM design and record upstream provenance."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET

COMMIT = "e337f3b177b4b53abff70b31d1695a7b66cc6d2e"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/scr_phybot_mjab/assets/toddlerbot/upstream"
BASE = f"https://raw.githubusercontent.com/hshi74/toddlerbot/{COMMIT}/"
DESCRIPTION = "toddlerbot/descriptions/toddlerbot_2xm/"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    xml = urlopen(BASE + DESCRIPTION + "toddlerbot_2xm.xml", timeout=60).read()
    mesh_names = [m.get("file") for m in ET.fromstring(xml).findall("./asset/mesh")]
    sources = {
        "toddlerbot_2xm.xml": DESCRIPTION + "toddlerbot_2xm.xml",
        "robot.yml": DESCRIPTION + "robot.yml",
        "default.yml": "toddlerbot/descriptions/default.yml",
        "LICENSE": "LICENSE",
        "README.md": "README.md",
        **{f"assets/{n}": DESCRIPTION + f"assets/{n}" for n in mesh_names},
    }

    def fetch(item):
        target, source = item
        data = xml if target == "toddlerbot_2xm.xml" else urlopen(BASE + source, timeout=60).read()
        path = DEST / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return target, {"source": source, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = dict(pool.map(fetch, sources.items()))
    (DEST.parent / "provenance.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/hshi74/toddlerbot",
                "commit": COMMIT,
                "code_license": "MIT",
                "design_license": "CC BY-NC-SA 4.0 (upstream README)",
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Imported {len(files)} files ({sum(v['bytes'] for v in files.values()) / 1e6:.1f} MB) into {DEST}")


if __name__ == "__main__":
    main()
