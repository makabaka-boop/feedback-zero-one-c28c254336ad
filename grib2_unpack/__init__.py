"""Strict decoder for the GRIB2 Template 5.3/7.3 subset used by this project.

The input is a concatenation of Section 5, Section 6 and Section 7 triples.
No other GRIB sections and no third-party GRIB decoder are used.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

__all__ = ["GribParseError", "unpack_stream"]


class GribParseError(Exception):
    """An error whose ``offset`` is a zero-based byte position in the stream."""

    def __init__(self, message: str, offset: int, parameter: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.offset = offset
        self.parameter = parameter

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        if self.parameter:
            return f"{self.parameter} at byte {self.offset}: {self.message}"
        return f"byte {self.offset}: {self.message}"


@dataclass(frozen=True)
class _Section5:
    start: int
    length: int
    number_of_values: int
    template_number: int
    reference_bits: bytes
    reference_value: Fraction
    binary_scale: int
    decimal_scale: int
    bits_per_value: int
    original_type: int
    group_splitting: int
    missing_management: int
    number_of_groups: int
    group_width_reference: int
    group_width_bits: int
    group_length_reference: int
    group_length_increment: int
    last_group_length: int
    group_length_bits: int
    order: int
    extra_octets: int


class _BitReader:
    """MSB-first bit reader; padding bits are explicitly checked by callers."""

    def __init__(self, data: bytes, start: int, end: int):
        self.data = data
        self.byte_pos = start
        self.bit_pos = 0
        self.end = end

    @property
    def absolute_bit(self) -> int:
        return self.byte_pos * 8 + self.bit_pos

    def read(self, bits: int) -> int:
        if bits < 0 or self.remaining_bits < bits:
            raise EOFError("truncated bit field")
        value = 0
        while bits:
            available = 8 - self.bit_pos
            take = min(bits, available)
            unused_in_taken = available - take
            chunk = (self.data[self.byte_pos] >> unused_in_taken) & ((1 << take) - 1)
            value = (value << take) | chunk
            self.bit_pos += take
            bits -= take
            if self.bit_pos == 8:
                self.byte_pos += 1
                self.bit_pos = 0
        return value

    @property
    def remaining_bits(self) -> int:
        return self.end * 8 - self.absolute_bit

    def align_to_octet(self, section_start: int) -> None:
        if self.bit_pos:
            padding = self.read(8 - self.bit_pos)
            if padding:
                raise GribParseError("descriptor padding bits must be zero", section_start)
            # read() leaves the reader positioned at the next octet.

    def require_empty(self, section_start: int) -> None:
        if self.remaining_bits:
            padding = self.read(self.remaining_bits)
            if padding:
                raise GribParseError("data section padding bits must be zero", section_start)


def _u32(data: bytes, pos: int, section_start: int) -> int:
    if pos + 4 > len(data):
        raise GribParseError("truncated four-octet unsigned field", section_start)
    return int.from_bytes(data[pos : pos + 4], "big", signed=False)


def _u16(data: bytes, pos: int, section_start: int) -> int:
    if pos + 2 > len(data):
        raise GribParseError("truncated two-octet unsigned field", section_start)
    return int.from_bytes(data[pos : pos + 2], "big", signed=False)


def _s16(data: bytes, pos: int, section_start: int) -> int:
    value = _u16(data, pos, section_start)
    return value - 0x10000 if value & 0x8000 else value


def _u8(data: bytes, pos: int, section_start: int) -> int:
    if pos >= len(data):
        raise GribParseError("truncated one-octet field", section_start)
    return data[pos]


def _section_length_and_number(data: bytes, start: int, expected_number: int) -> Tuple[int, int]:
    if start + 5 > len(data):
        raise GribParseError(f"Section {expected_number} is truncated", start)
    length = _u32(data, start, start)
    if length < 5:
        raise GribParseError(f"Section {expected_number} length must be at least 5", start)
    if start + length > len(data):
        raise GribParseError(f"Section {expected_number} extends beyond the stream", start)
    number = data[start + 4]
    if number != expected_number:
        raise GribParseError(
            f"expected Section {expected_number}, found Section {number}", start
        )
    return length, number


def _expect_section(data: bytes, start: int, expected_number: int) -> None:
    if start >= len(data):
        raise GribParseError(f"Section {expected_number} is missing", len(data))
    if start + 5 > len(data) or data[start + 4] != expected_number:
        found = data[start + 4] if start + 4 < len(data) else "truncated section"
        raise GribParseError(
            f"expected Section {expected_number}, found {found}", start
        )


def _read_section5(data: bytes, start: int) -> _Section5:
    length, _ = _section_length_and_number(data, start, 5)
    if length != 49:
        raise GribParseError(
            "Template 5.3 Section 5 has a fixed 49-octet length in this subset", start
        )

    number_of_values = _u32(data, start + 5, start)
    template_number = _u16(data, start + 9, start)
    if template_number != 3:
        raise GribParseError(
            f"only Data Representation Template 5.3 is accepted (got {template_number})",
            start,
        )

    reference_bits = data[start + 11 : start + 15]
    if len(reference_bits) != 4:
        raise GribParseError("truncated reference value R", start)
    reference_float = struct.unpack(">f", reference_bits)[0]
    if not math.isfinite(reference_float):
        raise GribParseError("reference value R must be finite", start)
    # Fraction(float) gives the exact IEEE-754 single precision rational value.
    reference_value = Fraction(reference_float)

    binary_scale = _s16(data, start + 15, start)
    decimal_scale = _s16(data, start + 17, start)
    bits_per_value = _u8(data, start + 19, start)
    original_type = _u8(data, start + 20, start)
    group_splitting = _u8(data, start + 21, start)
    missing_management = _u8(data, start + 22, start)

    # Substitute values are deliberately not used. They still have to be read
    # so the remaining fixed template offsets are known.
    primary_substitute = _u32(data, start + 23, start)
    secondary_substitute = _u32(data, start + 27, start)
    if missing_management == 0:
        # With no explicit missing management, all-one packed values are ordinary
        # data rather than sentinels. Keeping substitutes avoids ambiguous encodings.
        if primary_substitute != 0xFFFFFFFF or secondary_substitute != 0xFFFFFFFF:
            raise GribParseError(
                "missing substitute values must be encoded all-ones when management is 0",
                start,
            )
    if missing_management in (1, 2) and primary_substitute == 0xFFFFFFFF:
        raise GribParseError(
            "primary missing-value substitute must not be the all-ones value", start
        )
    if missing_management == 2 and secondary_substitute == 0xFFFFFFFE:
        raise GribParseError(
            "secondary missing-value substitute must not be its sentinel", start
        )

    number_of_groups = _u32(data, start + 31, start)
    group_width_reference = _u8(data, start + 35, start)
    group_width_bits = _u8(data, start + 36, start)
    group_length_reference = _u32(data, start + 37, start)
    group_length_increment = _u8(data, start + 41, start)
    last_group_length = _u32(data, start + 42, start)
    group_length_bits = _u8(data, start + 46, start)
    order = _u8(data, start + 47, start)
    extra_octets = _u8(data, start + 48, start)

    if original_type != 0:
        raise GribParseError("only type of original field values 0 is accepted", start)
    if group_splitting != 1:
        raise GribParseError("only group splitting method 1 is accepted", start)
    if missing_management not in (0, 1, 2):
        raise GribParseError("missing value management must be 0, 1 or 2", start)
    if order not in (1, 2):
        raise GribParseError("spatial differencing order must be 1 or 2", start)
    if missing_management == 2 and bits_per_value < 2:
        raise GribParseError(
            "secondary missing management needs distinct primary and secondary group-reference sentinels",
            start,
        )

    # These four values are required to be positive by the stated decoder subset.
    for name, value in (
        ("numberOfGroupsOfDataValues", number_of_groups),
        ("bitsPerValue", bits_per_value),
        ("numberOfBitsForScaledGroupLengths", group_length_bits),
        ("numberOfOctetsExtraDescriptors", extra_octets),
    ):
        if value <= 0:
            raise GribParseError(f"{name} must be positive", start, name)

    # The group width descriptor is a packed array; a zero field width would make
    # its byte-aligned descriptor physically absent. It is outside this subset.
    if group_width_bits <= 0:
        raise GribParseError(
            "numberOfBitsUsedForTheGroupWidths must be positive",
            start,
            "numberOfBitsUsedForTheGroupWidths",
        )
    if group_length_reference <= 0 or group_length_increment <= 0 or last_group_length <= 0:
        raise GribParseError("group length descriptors must describe positive lengths", start)

    # Avoid accidental reliance on Python-size integers and keep GRIB 32-bit ranges.
    if bits_per_value > 32 or group_width_bits > 32 or group_length_bits > 32 or extra_octets > 4:
        raise GribParseError("descriptor width exceeds the supported 32-bit range", start)

    return _Section5(
        start=start,
        length=length,
        number_of_values=number_of_values,
        template_number=template_number,
        reference_bits=reference_bits,
        reference_value=reference_value,
        binary_scale=binary_scale,
        decimal_scale=decimal_scale,
        bits_per_value=bits_per_value,
        original_type=original_type,
        group_splitting=group_splitting,
        missing_management=missing_management,
        number_of_groups=number_of_groups,
        group_width_reference=group_width_reference,
        group_width_bits=group_width_bits,
        group_length_reference=group_length_reference,
        group_length_increment=group_length_increment,
        last_group_length=last_group_length,
        group_length_bits=group_length_bits,
        order=order,
        extra_octets=extra_octets,
    )


def _read_section6(
    data: bytes,
    start: int,
    n: int,
    remembered_bitmap: Optional[Tuple[int, ...]],
) -> Tuple[List[int], Optional[Tuple[int, ...]], int]:
    length, _ = _section_length_and_number(data, start, 6)
    indicator = _u8(data, start + 5, start)
    if indicator not in (0, 254, 255):
        raise GribParseError("bit-map indicator must be 0, 254 or 255", start)

    if indicator == 0:
        section_bit_capacity = (length - 6) * 8
        if section_bit_capacity < n:
            raise GribParseError(
                f"explicit bit-map has capacity for {section_bit_capacity} bits but N={n}",
                start,
                "N",
            )
        reader = _BitReader(data, start + 6, start + length)
        bitmap = [reader.read(1) for _ in range(n)]
        reader.require_empty(start)
        remembered = tuple(bitmap)
        return bitmap, remembered, length

    if length != 6:
        raise GribParseError("Section 6 without an explicit bitmap must be six octets", start)

    if indicator == 254:
        if remembered_bitmap is None:
            raise GribParseError("bit-map indicator 254 has no preceding explicit bit-map", start)
        return list(remembered_bitmap), remembered_bitmap, length

    # 255: every grid point has a value. It must not clear the remembered bitmap.
    return [1] * n, remembered_bitmap, length


def _read_aligned_array(
    section_start: int,
    reader: _BitReader,
    count: int,
    width: int,
) -> List[int]:
    values: List[int] = []
    for _ in range(count):
        values.append(reader.read(width))
    reader.align_to_octet(section_start)
    return values


def _decode_signed_sign_magnitude(bits: int, value: int, section_start: int) -> int:
    if bits <= 0:
        raise GribParseError("sign-magnitude field must have a sign bit", section_start)
    magnitude_bits = bits - 1
    magnitude = value & ((1 << magnitude_bits) - 1)
    return -magnitude if value >> magnitude_bits else magnitude


def _read_section7(
    data: bytes,
    start: int,
    descriptor: _Section5,
    bitmap: Sequence[int],
) -> List[object]:
    length, _ = _section_length_and_number(data, start, 7)
    if length < 6:
        raise GribParseError("Section 7 must contain data", start)

    d = descriptor
    reader = _BitReader(data, start + 5, start + length)

    # Template 7.3 extra descriptors: n initial values followed by the minimum.
    # They immediately precede the group-reference array; the combined block ends
    # on an octet boundary after those group references.
    extra_width = d.extra_octets * 8
    initials = [reader.read(extra_width) for _ in range(d.order)]
    minimum_raw = reader.read(extra_width)
    minimum_difference = _decode_signed_sign_magnitude(
        extra_width, minimum_raw, start
    )

    references = _read_aligned_array(
        start, reader, d.number_of_groups, d.bits_per_value
    )
    packed_widths = _read_aligned_array(
        start, reader, d.number_of_groups, d.group_width_bits
    )
    scaled_lengths = _read_aligned_array(
        start, reader, d.number_of_groups, d.group_length_bits
    )

    widths = [w + d.group_width_reference for w in packed_widths]
    if any(width > 32 for width in widths):
        raise GribParseError("decoded group width exceeds 32 bits", start)
    lengths: List[int] = []
    for index, scaled in enumerate(scaled_lengths):
        if index == d.number_of_groups - 1:
            # The last scaled length is still consumed, but the true value wins.
            lengths.append(d.last_group_length)
        else:
            increment = scaled * d.group_length_increment
            if increment > 0xFFFFFFFF or d.group_length_reference + increment > 0xFFFFFFFF:
                raise GribParseError("decoded group length exceeds its 32-bit range", start)
            lengths.append(d.group_length_reference + increment)

    if sum(lengths) != d.number_of_values:
        raise GribParseError(
            "sum of decoded group lengths must equal numberOfValues", start
        )
    valid_count = sum(1 for bit in bitmap if bit == 1)
    if d.number_of_values != valid_count:
        raise GribParseError(
            "numberOfValues must equal the number of ones in the effective bit-map",
            start,
            "numberOfValues",
        )

    slots: List[object] = []

    for group_ref, width, group_length in zip(references, widths, lengths):
        if width == 0:
            if d.missing_management == 1:
                sentinel = (1 << d.bits_per_value) - 1
                if group_ref == sentinel:
                    slots.extend(["primary"] * group_length)
                    continue
            elif d.missing_management == 2:
                primary = (1 << d.bits_per_value) - 1
                secondary = primary - 1
                if group_ref == primary:
                    slots.extend(["primary"] * group_length)
                    continue
                if group_ref == secondary:
                    slots.extend(["secondary"] * group_length)
                    continue
            # A normal constant group consumes no packed increment bits.
            slots.extend([group_ref] * group_length)
            continue

        if d.missing_management == 1:
            primary_sentinel_width = (1 << width) - 1
        elif d.missing_management == 2:
            primary_sentinel_width = (1 << width) - 1
            secondary_sentinel_width = primary_sentinel_width - 1
            if width < 2:
                raise GribParseError(
                    "secondary missing management requires at least two packed-value bits",
                    start,
                )

        for _ in range(group_length):
            packed = reader.read(width)
            if d.missing_management == 1 and packed == primary_sentinel_width:
                slots.append("primary")
            elif d.missing_management == 2 and packed == primary_sentinel_width:
                slots.append("primary")
            elif d.missing_management == 2 and packed == secondary_sentinel_width:
                slots.append("secondary")
            else:
                value = group_ref + packed
                if value < 0 or value > 0xFFFFFFFF:
                    raise GribParseError("decoded scaled value exceeds its 32-bit range", start)
                slots.append(value)

    if len(slots) != d.number_of_values:
        raise GribParseError("decoded slot count does not match numberOfValues", start)
    reader.require_empty(start)

    non_missing_count = sum(isinstance(value, int) for value in slots)
    if non_missing_count < d.order:
        raise GribParseError(
            "number of non-missing values must be at least the differencing order",
            start,
        )

    # The first n non-missing packed slots are dummy values. Replace them with
    # the explicit initial scaled values, without changing the missing stream.
    initial_index = 0
    for index, value in enumerate(slots):
        if isinstance(value, int):
            slots[index] = initials[initial_index]
            initial_index += 1
            if initial_index == d.order:
                break

    _undo_spatial_differencing(slots, d.order, minimum_difference)
    return slots


def _undo_spatial_differencing(slots: List[object], order: int, bias: int) -> None:
    if order == 1:
        last: Optional[int] = None
        for index, value in enumerate(slots):
            if not isinstance(value, int):
                continue
            if last is None:
                last = value
            else:
                value = value + last + bias
                slots[index] = value
                last = value
        return

    penultimate: Optional[int] = None
    last = None
    for index, value in enumerate(slots):
        if not isinstance(value, int):
            continue
        if penultimate is None:
            penultimate = value
        elif last is None:
            last = value
        else:
            # Both previous operands are known because missing slots skip recurrence.
            value = value + bias + 2 * last - penultimate  # type: ignore[operator]
            slots[index] = value
            penultimate, last = last, value


def _scale_value(value: int, descriptor: _Section5) -> Fraction:
    # (R + X * 2^E) * 10^(-D), computed exactly as a rational number.
    result = descriptor.reference_value + Fraction(value) * Fraction(2) ** descriptor.binary_scale
    if descriptor.decimal_scale >= 0:
        result = result / (Fraction(10) ** descriptor.decimal_scale)
    else:
        result = result * (Fraction(10) ** (-descriptor.decimal_scale))
    return result


def _format_fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def unpack_stream(n: int, stream: bytes) -> List[List[str]]:
    """Decode all fields in a concatenated 5/6/7 stream.

    Args:
        n: Required number of grid points for every output field.
        stream: Binary concatenation of Section 5 + Section 6 + Section 7 triples.

    Returns:
        A two-dimensional JSON-compatible list. Missing values use the strings
        ``"bitmap"``, ``"primary"`` and ``"secondary"``.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise GribParseError("N must be a positive integer", 0, "N")
    if not isinstance(stream, (bytes, bytearray, memoryview)):
        raise GribParseError("stream must be bytes-like", 0)
    stream = bytes(stream)
    if not stream:
        return []

    fields: List[List[str]] = []
    pos = 0
    remembered_bitmap: Optional[Tuple[int, ...]] = None

    while pos < len(stream):
        field_start = pos

        section5 = _read_section5(stream, pos)
        if section5.start != field_start:
            raise GribParseError("invalid Section 5", field_start)
        pos += section5.length
        _expect_section(stream, pos, 6)

        try:
            bitmap, remembered_bitmap, section6_length = _read_section6(
                stream, pos, n, remembered_bitmap
            )
        except EOFError as exc:
            raise GribParseError(str(exc), pos) from exc
        pos += section6_length
        _expect_section(stream, pos, 7)

        if len(bitmap) != n:
            raise GribParseError(
                f"effective bit-map has {len(bitmap)} bits but N={n}",
                section5.start,
                "N",
            )

        section7_start = pos
        try:
            scaled_slots = _read_section7(stream, pos, section5, bitmap)
        except EOFError as exc:
            raise GribParseError(str(exc), section7_start) from exc
        section7_length = _u32(stream, section7_start, section7_start)
        pos += section7_length

        field: List[str] = []
        slot_index = 0
        for bit in bitmap:
            if bit == 0:
                field.append("bitmap")
            else:
                scaled = scaled_slots[slot_index]
                slot_index += 1
                if isinstance(scaled, str):
                    field.append(scaled)
                else:
                    field.append(_format_fraction(_scale_value(scaled, section5)))
        if slot_index != len(scaled_slots):
            raise GribParseError("not all packed values were consumed by the bit-map", pos)
        fields.append(field)

    if pos != len(stream):
        raise GribParseError("unexpected trailing bytes", pos)
    return fields
