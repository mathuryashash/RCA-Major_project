"""Pull readable text out of a PDF without poppler.

Only good enough to identify which source produced a file and to spot layout
symptoms; it is not a general extractor.
"""

import re
import sys
import zlib
from pathlib import Path


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "major_project_paper.pdf")
    data = target.read_bytes()

    chunks = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            chunks.append(zlib.decompress(match.group(1)).decode("latin-1"))
        except Exception:                       # noqa: BLE001 - not all streams are text
            continue

    content = "\n".join(chunks)
    shown = re.findall(r"\(((?:[^()\\]|\\.)*)\)", content)
    plain = re.sub(r"\s+", " ", " ".join(shown))

    print(f"streams decoded : {len(chunks)}")
    print(f"characters      : {len(plain)}")
    print()
    print(plain[: int(sys.argv[2]) if len(sys.argv) > 2 else 1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
