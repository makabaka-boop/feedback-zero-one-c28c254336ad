"""Optional numeric cross-check against ecCodes.

This script is intentionally optional. The repository's strict parser is fully
exercised without ecCodes by ``python3 -I -m unittest discover``. If the
``eccodes`` Python module and its native library are installed, this script
creates complete in-memory GRIB2 messages, asks ecCodes for their values, and
compares only the numeric intersection accepted by both implementations.

The independent strict-rejection and missing-classification fixtures are in
``tests/fixtures`` and do not depend on ecCodes.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grib2_unpack import unpack_stream  # noqa: E402
from tests.build_fixtures import Section5Options, build_field  # noqa: E402

try:
    import eccodes  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"SKIP: ecCodes is not available ({exc!r})")
    raise SystemExit(0)


def make_case() -> tuple[int, bytes, List[float]]:
    n = 6
    options = Section5Options(
        number_of_values=6,
        reference=0.25,
        binary_scale=1,
        decimal_scale=1,
        bits_per_value=8,
        number_of_groups=2,
        group_width_reference=2,
        group_width_bits=2,
        group_length_reference=3,
        last_group_length=3,
        group_length_bits=1,
        order=1,
        extra_octets=1,
        missing_management=0,
    )
    # Initial X=2; differences 1, 2, 3, 4, 5, with bias=1.
    # Reconstructed X: 2,4,7,11,16,22.
    stream_567 = build_field(
        n=n,
        bitmap=None,
        options=options,
        initials=(2,),
        minimum_difference=1,
        references=(0, 0),
        widths=(2, 2),
        scaled_lengths=(0, 0),
        groups=[(0, 1, 2), (3, 4, 5)],
    )
    expected_x = [2, 4, 7, 11, 16, 22]
    expected = [
        float((Fraction(0.25) + Fraction(x) * Fraction(2)) / Fraction(10))
        for x in expected_x
    ]
    return n, stream_567, expected


def wrap_minimal_message(sections_567: bytes) -> bytes:
    # Section 0: indicator, 16 octets.
    section0 = b"GRIB" + bytes([0, 2]) + (16).to_bytes(8, "big")
    # Section 1: fixed 21-octet identification section.
    section1_body = (
        (0).to_bytes(2, "big")  # centre
        + (0).to_bytes(2, "big")  # sub-centre
        + bytes([2, 0, 1])  # table versions and reference-time significance
        + (2026).to_bytes(2, "big")
        + bytes([1, 1, 0, 0, 0, 0, 0])
    )
    assert len(section1_body) == 16
    section1 = (len(section1_body) + 5).to_bytes(4, "big") + bytes([1]) + section1_body
    # Section 3 and 4 use one point and a grid point template.
    section3_body = (
        (0).to_bytes(4, "big")  # source
        + (1).to_bytes(4, "big")  # number of data points
        + bytes([0, 0])  # optional point list octets and interpretation
        + (0).to_bytes(2, "big")  # grid definition template number
    )
    section3 = (len(section3_body) + 5).to_bytes(4, "big") + bytes([3]) + section3_body
    section4_body = (
        (0).to_bytes(2, "big")  # no coordinate values
        + (0).to_bytes(2, "big")  # product definition template 0
    )
    section4 = (len(section4_body) + 5).to_bytes(4, "big") + bytes([4]) + section4_body
    section8 = b"7777"
    return section0 + section1 + section3 + section4 + sections_567 + section8


def main() -> int:
    n, stream_567, expected = make_case()
    ours = unpack_stream(n, stream_567)
    ours_floats = [float(Fraction(value)) for field in ours for value in field]
    if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(ours_floats, expected)):
        raise AssertionError(("internal numeric mismatch", ours_floats, expected))

    message = wrap_minimal_message(sections_567=stream_567)
    handle = eccodes.codes_new_from_message(message)
    try:
        count = eccodes.codes_get(handle, "numberOfDataPoints")
        if count != n:
            raise AssertionError(("ecCodes point count", count, n))
        eccodes_values = eccodes.codes_get_array(handle, "values")
    finally:
        eccodes.codes_release(handle)

    if len(eccodes_values) != n:
        raise AssertionError(("ecCodes value count", len(eccodes_values), n))
    for index, (ours_value, eccodes_value, exact_value) in enumerate(
        zip(ours_floats, eccodes_values, expected)
    ):
        if not math.isclose(ours_value, eccodes_value, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError((index, ours_value, eccodes_value, exact_value))

    print(f"OK: {n} numeric values agree with ecCodes: {eccodes_values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
