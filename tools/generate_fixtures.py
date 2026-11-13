"""Generate static, hand-built byte fixtures and expected JSON outputs.

Run from the repository root:

    python3 -I tools/generate_fixtures.py

The fixtures contain raw concatenated GRIB2 Sections 5, 6 and 7 only. They are
built with the project's independent byte writer, not with a GRIB decoder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.build_fixtures import Section5Options, build_field, section5, section6, section7

FIXTURE_DIR = ROOT / "tests" / "fixtures"


def write_case(name: str, stream: bytes, expected: object) -> None:
    (FIXTURE_DIR / f"{name}.bin").write_bytes(stream)
    (FIXTURE_DIR / f"{name}.expected.json").write_text(
        json.dumps(expected, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    n = 4
    sample = Section5Options(
        number_of_values=3,
        bits_per_value=4,
        number_of_groups=3,
        group_width_reference=1,
        group_width_bits=1,
        group_length_reference=1,
        group_length_increment=1,
        last_group_length=1,
        group_length_bits=1,
        order=1,
        extra_octets=1,
    )
    write_case(
        "spec_first_order_bitmap",
        build_field(
            n=n,
            bitmap=(1, 0, 1, 1),
            options=sample,
            initials=(4,),
            minimum_difference=1,
            references=(0, 0, 0),
            widths=(1, 2, 2),
            scaled_lengths=(0, 0, 0),
            groups=[(0,), (2,), (1,)],
        ),
        [["4/1", "bitmap", "7/1", "9/1"]],
    )

    n = 5
    second_order = Section5Options(
        number_of_values=5,
        bits_per_value=5,
        missing_management=1,
        primary_substitute=1,
        number_of_groups=1,
        group_width_reference=0,
        group_width_bits=3,
        group_length_reference=5,
        last_group_length=5,
        order=2,
        extra_octets=1,
    )
    write_case(
        "second_order_primary_missing",
        build_field(
            n=n,
            bitmap=None,
            options=second_order,
            initials=(4, 6),
            minimum_difference=0,
            references=(0,),
            widths=(4,),
            scaled_lengths=(0,),
            groups=[(0, 0, 5, "primary", 10)],
        ),
        [["4/1", "6/1", "13/1", "primary", "30/1"]],
    )

    n = 4
    both_missing = Section5Options(
        number_of_values=4,
        bits_per_value=2,
        missing_management=2,
        primary_substitute=1,
        secondary_substitute=2,
        number_of_groups=2,
        group_width_reference=0,
        group_width_bits=2,
        group_length_reference=2,
        last_group_length=2,
        group_length_bits=1,
        order=1,
        extra_octets=1,
    )
    write_case(
        "primary_secondary_missing",
        build_field(
            n=n,
            bitmap=(1, 1, 1, 1),
            options=both_missing,
            initials=(9,),
            minimum_difference=0,
            references=(0, 3),
            widths=(2, 0),
            scaled_lengths=(0, 0),
            groups=[(0, "secondary"), ()],
        ),
        [["9/1", "secondary", "primary", "primary"]],
    )

    # Invalid: bitmap indicator 254 appears before any explicit bitmap.
    bad = section5(sample) + section6(None, 4)
    bad = bad[:49] + bytes([6, 0, 0, 0, 6, 254])
    write_case("reject_bitmap_reuse_without_memory", bad, None)

    # Invalid: the final Section 7 padding bit is one.
    n = 1
    constant = Section5Options(
        number_of_values=1,
        bits_per_value=4,
        number_of_groups=1,
        group_width_reference=1,
        group_length_reference=1,
        last_group_length=1,
        order=1,
    )
    bad_padding = (
        section5(constant)
        + section6(None, n)
        + section7(
            initials=(1,),
            minimum_difference=0,
            references=(0,),
            widths=(1,),
            scaled_lengths=(0,),
            groups=[(1,)],
            options=constant,
            bad_padding=True,
        )
    )
    write_case("reject_nonzero_padding", bad_padding, None)


if __name__ == "__main__":
    main()
