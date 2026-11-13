from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from grib2_unpack import GribParseError, unpack_stream
from tests.build_fixtures import (
    Section5Options,
    build_field,
    section5,
    section6,
    section6_reuse,
    section7,
)

ROOT = Path(__file__).resolve().parents[1]


class StaticFixtureTests(unittest.TestCase):
    fixture_dir = ROOT / "tests" / "fixtures"

    def test_generated_valid_byte_fixtures(self):
        cases = {
            4: [
                "spec_first_order_bitmap",
                "primary_secondary_missing",
            ],
            5: ["second_order_primary_missing"],
        }
        for n, names in cases.items():
            for name in names:
                with self.subTest(name=name):
                    stream = (self.fixture_dir / f"{name}.bin").read_bytes()
                    expected = json.loads(
                        (self.fixture_dir / f"{name}.expected.json").read_text()
                    )
                    self.assertEqual(unpack_stream(n, stream), expected)

    def test_generated_rejection_fixtures_report_offsets(self):
        stream = (
            self.fixture_dir / "reject_bitmap_reuse_without_memory.bin"
        ).read_bytes()
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(4, stream)
        self.assertEqual(caught.exception.offset, 49)

        stream = (self.fixture_dir / "reject_nonzero_padding.bin").read_bytes()
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(1, stream)
        self.assertEqual(caught.exception.offset, 55)


class UnpackAcceptanceTests(unittest.TestCase):
    def test_specification_first_order_bitmap_example(self):
        n = 4
        options = Section5Options(
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
        stream = build_field(
            n=n,
            bitmap=(1, 0, 1, 1),
            options=options,
            initials=(4,),
            minimum_difference=1,
            references=(0, 0, 0),
            widths=(1, 2, 2),
            scaled_lengths=(0, 0, 0),
            groups=[(0,), (2,), (1,)],
        )
        self.assertEqual(
            unpack_stream(n, stream), [["4/1", "bitmap", "7/1", "9/1"]]
        )

    def test_exact_rational_scaling_and_reduced_denominator(self):
        # R=0.5 (exact binary32), E=1, D=2 and X=3:
        # (0.5 + 3*2) / 100 = 6.5/100 = 13/200.
        n = 1
        options = Section5Options(
            number_of_values=1,
            reference=0.5,
            binary_scale=1,
            decimal_scale=2,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=0,
            group_width_bits=2,
            last_group_length=1,
            order=1,
            extra_octets=1,
        )
        stream = build_field(
            n=n,
            bitmap=None,
            options=options,
            initials=(3,),
            minimum_difference=0,
            references=(0,),
            widths=(2,),
            scaled_lengths=(0,),
            groups=[(3,)],
        )
        self.assertEqual(unpack_stream(n, stream), [["13/200"]])
        self.assertEqual(
            (Fraction(1, 2) + Fraction(3) * Fraction(2)) / Fraction(100),
            Fraction(13, 200),
        )

    def test_negative_scale_factors_remain_exact_rational(self):
        n = 1
        options = Section5Options(
            number_of_values=1,
            reference=0.0,
            binary_scale=-1,
            decimal_scale=-1,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=0,
            group_width_bits=2,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        # (0 + 3 * 2^-1) * 10^1 = 3/2 * 10 = 15/1.
        stream = build_field(
            n=n,
            bitmap=None,
            options=options,
            initials=(3,),
            minimum_difference=0,
            references=(0,),
            widths=(2,),
            scaled_lengths=(0,),
            groups=[(3,)],
        )
        self.assertEqual(unpack_stream(n, stream), [["15/1"]])

    def test_negative_minimum_uses_sign_magnitude(self):
        n = 3
        options = Section5Options(
            number_of_values=3,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=2,
            group_width_bits=1,
            group_length_reference=3,
            last_group_length=3,
            order=1,
            extra_octets=1,
        )
        # Initial 10; packed increments including dummy: [0, 3, 2].
        # bias=-2 => values 10, 10+(3-2)=11, 11+(2-2)=11.
        stream = build_field(
            n=n,
            bitmap=None,
            options=options,
            initials=(10,),
            minimum_difference=-2,
            references=(0,),
            widths=(2,),
            scaled_lengths=(0,),
            groups=[(0, 3, 2)],
        )
        self.assertEqual(unpack_stream(n, stream), [["10/1", "11/1", "11/1"]])

    def test_second_order_reconstruction_skips_missing_slots(self):
        n = 5
        options = Section5Options(
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
        # Initials 4,6; h values [dummy,dummy,5,missing,10].
        # bias=0: third=5+2*6-4=13; fifth=10+2*13-6=30.
        stream = build_field(
            n=n,
            bitmap=None,
            options=options,
            initials=(4, 6),
            minimum_difference=0,
            references=(0,),
            widths=(4,),
            scaled_lengths=(0,),
            groups=[(0, 0, 5, "primary", 10)],
        )
        self.assertEqual(
            unpack_stream(n, stream),
            [["4/1", "6/1", "13/1", "primary", "30/1"]],
        )

    def test_primary_secondary_and_constant_missing_groups(self):
        n = 4
        options = Section5Options(
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
        # Group 0: width 2, sentinels 3=primary and 2=secondary.
        # Group 1: width 0, reference 3 means an all-primary constant group.
        # Four slots: initial 9 replaces the first dummy, then secondary, primary,
        # primary. Only one non-missing value is required by first-order differencing.
        stream = build_field(
            n=n,
            bitmap=(1, 1, 1, 1),
            options=options,
            initials=(9,),
            minimum_difference=0,
            references=(0, 3),
            widths=(2, 0),
            scaled_lengths=(0, 0),
            groups=[(0, "secondary"), ()],
        )
        self.assertEqual(
            unpack_stream(n, stream),
            [["9/1", "secondary", "primary", "primary"]],
        )

    def test_bitmap_reuse_and_255_does_not_clear_memory(self):
        n = 3
        bitmap = (1, 0, 1)
        first_options = Section5Options(
            number_of_values=2,
            bits_per_value=4,
            number_of_groups=2,
            group_width_reference=1,
            last_group_length=1,
            order=1,
        )
        first = build_field(
            n=n,
            bitmap=bitmap,
            options=first_options,
            initials=(1,),
            minimum_difference=1,
            references=(0, 0),
            widths=(1, 1),
            scaled_lengths=(0, 0),
            groups=[(0,), (1,)],
        )

        all_present_options = Section5Options(
            number_of_values=3,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=2,
            group_length_reference=3,
            last_group_length=3,
            order=1,
        )
        middle = (
            section5(all_present_options)
            + section6(None, n)
            + section7(
                initials=(7,),
                minimum_difference=0,
                references=(0,),
                widths=(2,),
                scaled_lengths=(0,),
                groups=[(0, 1, 2)],
                options=all_present_options,
            )
        )

        reused = (
            section5(first_options)
            + section6_reuse()
            + section7(
                initials=(2,),
                minimum_difference=2,
                references=(0, 0),
                widths=(1, 1),
                scaled_lengths=(0, 0),
                groups=[(0,), (0,)],
                options=first_options,
            )
        )
        result = unpack_stream(n, first + middle + reused)
        self.assertEqual(
            result,
            [
                ["1/1", "bitmap", "3/1"],
                ["7/1", "8/1", "10/1"],
                ["2/1", "bitmap", "4/1"],
            ],
        )

    def test_empty_stream(self):
        self.assertEqual(unpack_stream(1, b""), [])

    def test_bytes_like_inputs_are_accepted(self):
        stream = (ROOT / "tests" / "fixtures" / "spec_first_order_bitmap.bin").read_bytes()
        expected = [["4/1", "bitmap", "7/1", "9/1"]]
        self.assertEqual(unpack_stream(4, bytearray(stream)), expected)
        self.assertEqual(unpack_stream(4, memoryview(stream)), expected)

    def test_descriptor_padding_must_be_zero(self):
        n = 1
        options = Section5Options(
            number_of_values=1,
            bits_per_value=3,
            number_of_groups=1,
            group_width_reference=1,
            group_width_bits=1,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        good = build_field(
            n=n,
            bitmap=None,
            options=options,
            initials=(1,),
            minimum_difference=0,
            references=(0,),
            widths=(1,),
            scaled_lengths=(0,),
            groups=[(1,)],
        )
        section7_start = len(section5(options)) + len(section6(None, n))
        # Section 7 payload: three 8-bit extra/ref fields, one 1-bit width,
        # one 1-bit length, then packed data. Corrupt the width-array pad bit.
        bad = bytearray(good)
        payload_bit_offset = (section7_start + 5) * 8 + 3 * 8 + 1
        byte_index = payload_bit_offset // 8
        bad[byte_index] |= 1 << (7 - (payload_bit_offset % 8))
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, bytes(bad))
        self.assertEqual(caught.exception.offset, section7_start)

    def test_strict_padding_and_offset_rejection(self):
        n = 1
        options = Section5Options(
            number_of_values=1,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=1,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        good = build_field(
            n=n,
            bitmap=None,
            options=options,
            initials=(1,),
            minimum_difference=0,
            references=(0,),
            widths=(1,),
            scaled_lengths=(0,),
            groups=[(1,)],
        )
        section7_start = len(section5(options)) + len(section6(None, n))

        # Wrong section order: second triple starts with a Section 6.
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, good + section6(None, n))
        self.assertEqual(caught.exception.offset, len(good))

        # Nonzero final padding bit inside Section 7; error locates Section 7.
        bad_padding = (
            section5(options)
            + section6(None, n)
            + section7(
                initials=(1,),
                minimum_difference=0,
                references=(0,),
                widths=(1,),
                scaled_lengths=(0,),
                groups=[(1,)],
                options=options,
                bad_padding=True,
            )
        )
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, bad_padding)
        self.assertEqual(caught.exception.offset, section7_start)

        # Truncation reports the start of the section being parsed.
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, good[: section7_start + 2])
        self.assertEqual(caught.exception.offset, section7_start)

        # Truncation inside Section 7's bit stream still reports that section.
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, good[:-1])
        self.assertEqual(caught.exception.offset, section7_start)

        # A missing following section reports the file length.
        section_only = section5(options)
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, section_only)
        self.assertEqual(caught.exception.offset, len(section_only))

        # A malformed extra byte is diagnosed at the next section's start.
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, good + b"\x00")
        self.assertEqual(caught.exception.offset, len(good))

    def test_unsupported_values_report_section_offsets_and_parameter(self):
        n = 1
        options = Section5Options(
            number_of_values=1,
            bits_per_value=4,
            original_type=1,
            number_of_groups=1,
            group_width_reference=1,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        stream = section5(options)
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, stream)
        self.assertEqual(caught.exception.offset, 0)

        options.original_type = 0
        options.number_of_groups = 0
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, section5(options))
        self.assertEqual(caught.exception.parameter, "numberOfGroupsOfDataValues")

    def test_non_finite_reference_is_rejected(self):
        n = 1
        options = Section5Options(
            number_of_values=1,
            reference=float("nan"),
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=1,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        with self.assertRaisesRegex(GribParseError, "reference value R"):
            unpack_stream(n, section5(options))

    def test_unsupported_template_and_bitmap_codes_are_rejected(self):
        n = 1
        options = Section5Options(
            number_of_values=1,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=1,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        valid_section5 = section5(options)
        unsupported_template = bytearray(valid_section5)
        unsupported_template[9:11] = (2).to_bytes(2, "big")
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, bytes(unsupported_template))
        self.assertEqual(caught.exception.offset, 0)

        unsupported_bitmap = valid_section5 + bytes([6, 0, 0, 0, 6, 1])
        with self.assertRaises(GribParseError) as caught:
            unpack_stream(n, unsupported_bitmap)
        self.assertEqual(caught.exception.offset, len(valid_section5))

    def test_section5_positive_parameter_names_are_reported(self):
        n = 1
        defaults = Section5Options(
            number_of_values=1,
            bits_per_value=4,
            number_of_groups=1,
            group_width_reference=1,
            group_width_bits=1,
            group_length_reference=1,
            last_group_length=1,
            group_length_bits=1,
            order=1,
            extra_octets=1,
        )
        changes = {
            "bits_per_value": ("bitsPerValue", 0),
            "group_length_bits": ("numberOfBitsForScaledGroupLengths", 0),
            "extra_octets": ("numberOfOctetsExtraDescriptors", 0),
            "group_width_bits": ("numberOfBitsUsedForTheGroupWidths", 0),
        }
        for field_name, (parameter, bad_value) in changes.items():
            with self.subTest(parameter=parameter):
                options = Section5Options(**{**defaults.__dict__, field_name: bad_value})
                with self.assertRaises(GribParseError) as caught:
                    unpack_stream(n, section5(options))
                self.assertEqual(caught.exception.parameter, parameter)
                self.assertEqual(caught.exception.offset, 0)

    def test_cli_preserves_old_output_on_failure(self):
        n = 4
        options = Section5Options(
            number_of_values=3,
            bits_per_value=4,
            number_of_groups=3,
            group_width_reference=1,
            group_length_reference=1,
            last_group_length=1,
            order=1,
        )
        good = build_field(
            n=n,
            bitmap=(1, 0, 1, 1),
            options=options,
            initials=(4,),
            minimum_difference=1,
            references=(0, 0, 0),
            widths=(1, 2, 2),
            scaled_lengths=(0, 0, 0),
            groups=[(0,), (2,), (1,)],
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.bin"
            output_path = Path(directory) / "out.json"
            input_path.write_bytes(good)
            success = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grib2_unpack",
                    str(n),
                    str(input_path),
                    "-o",
                    str(output_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(
                json.loads(output_path.read_text()),
                [["4/1", "bitmap", "7/1", "9/1"]],
            )

            input_path.write_bytes(good + b"\x00")
            failure = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grib2_unpack",
                    str(n),
                    str(input_path),
                    "-o",
                    str(output_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(failure.returncode, 1)
            self.assertIn(f"byte {len(good)}", failure.stderr)
            self.assertEqual(
                json.loads(output_path.read_text()),
                [["4/1", "bitmap", "7/1", "9/1"]],
            )


if __name__ == "__main__":
    unittest.main()
