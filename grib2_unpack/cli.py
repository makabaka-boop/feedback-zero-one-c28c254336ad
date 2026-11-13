"""Command-line entry point for the strict GRIB2 5.3/7.3 unpacker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import GribParseError, unpack_stream


def _positive_integer(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("N must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("N must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grib2-unpack",
        description="Decode concatenated GRIB2 Sections 5, 6 and 7 (Templates 5.3/7.3).",
    )
    parser.add_argument("n", type=_positive_integer, help="number of grid points N")
    parser.add_argument("input", help="binary input stream (- for standard input)")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="JSON output file; it is replaced atomically only after successful decoding",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.input == "-":
            stream = sys.stdin.buffer.read()
        else:
            stream = Path(args.input).read_bytes()
        result = unpack_stream(args.n, stream)
    except GribParseError as exc:
        detail = f"{exc.parameter} " if exc.parameter else ""
        print(f"error: {detail}at byte {exc.offset}: {exc.message}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write and fsync a temporary file in the same directory, then atomically
    # replace the destination. A decoding failure never reaches this code, so an
    # existing output remains untouched.
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
