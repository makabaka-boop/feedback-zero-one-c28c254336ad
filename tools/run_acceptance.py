#!/usr/bin/env python3
"""Project acceptance runner: unit suite plus CLI and static-fixture checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    fixture = ROOT / "tests" / "fixtures" / "spec_first_order_bitmap.bin"
    expected = json.loads(
        (ROOT / "tests" / "fixtures" / "spec_first_order_bitmap.expected.json").read_text(
            encoding="utf-8"
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "grib2_unpack" / "__main__.py"),
                "4",
                str(fixture),
                "-o",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            print(completed.stderr, file=sys.stderr)
            return completed.returncode
        if json.loads(output.read_text(encoding="utf-8")) != expected:
            print("CLI output did not match the static fixture", file=sys.stderr)
            return 1

    print("ACCEPTANCE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
