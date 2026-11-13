"""Independent byte-fixture builders used by the acceptance tests.

This module writes raw GRIB2 Sections 5, 6 and 7 by hand. It deliberately does
not import or use the production decoder, and it uses no GRIB library.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


class BitWriter:
    def __init__(self) -> None:
        self.bytes = bytearray()
        self.current = 0
        self.used = 0

    def write(self, value: int, bits: int) -> None:
        if bits == 0:
            return
        if value < 0 or value >= (1 << bits):
            raise ValueError(f"value {value} does not fit in {bits} bits")
        remaining = bits
        while remaining:
            take = min(remaining, 8 - self.used)
            shift = remaining - take
            chunk = (value >> shift) & ((1 << take) - 1)
            self.current = (self.current << take) | chunk
            self.used += take
            remaining -= take
            if self.used == 8:
                self.bytes.append(self.current)
                self.current = 0
                self.used = 0

    def align(self) -> None:
        if self.used:
            self.write(0, 8 - self.used)

    def finish(self) -> bytes:
        self.align()
        return bytes(self.bytes)


def u8(value: int) -> bytes:
    return bytes([value])


def u16(value: int) -> bytes:
    return value.to_bytes(2, "big")


def s16(value: int) -> bytes:
    if value < 0:
        value += 1 << 16
    return u16(value)


def u32(value: int) -> bytes:
    return value.to_bytes(4, "big")


def ieee_f32(value: float) -> bytes:
    return struct.pack(">f", value)


def sign_magnitude(value: int, octets: int) -> int:
    bits = octets * 8
    magnitude_bits = bits - 1
    if abs(value) >= (1 << magnitude_bits):
        raise ValueError("sign-magnitude value too large")
    return (1 << magnitude_bits if value < 0 else 0) | abs(value)


@dataclass
class Section5Options:
    number_of_values: int
    reference: float = 0.0
    binary_scale: int = 0
    decimal_scale: int = 0
    bits_per_value: int = 4
    original_type: int = 0
    group_splitting: int = 1
    missing_management: int = 0
    primary_substitute: int = 0xFFFFFFFF
    secondary_substitute: int = 0xFFFFFFFF
    number_of_groups: int = 1
    group_width_reference: int = 0
    group_width_bits: int = 1
    group_length_reference: int = 1
    group_length_increment: int = 1
    last_group_length: int = 1
    group_length_bits: int = 1
    order: int = 1
    extra_octets: int = 1


def section5(options: Section5Options) -> bytes:
    template = b"".join(
        (
            ieee_f32(options.reference),
            s16(options.binary_scale),
            s16(options.decimal_scale),
            u8(options.bits_per_value),
            u8(options.original_type),
            u8(options.group_splitting),
            u8(options.missing_management),
            u32(options.primary_substitute),
            u32(options.secondary_substitute),
            u32(options.number_of_groups),
            u8(options.group_width_reference),
            u8(options.group_width_bits),
            u32(options.group_length_reference),
            u8(options.group_length_increment),
            u32(options.last_group_length),
            u8(options.group_length_bits),
            u8(options.order),
            u8(options.extra_octets),
        )
    )
    assert len(template) == 38
    length = 11 + len(template)
    return u32(length) + u8(5) + u32(options.number_of_values) + u16(3) + template


def section6(bitmap: Optional[Sequence[int]], n: int) -> bytes:
    if bitmap is None:
        return u32(6) + u8(6) + u8(255)
    if len(bitmap) != n:
        raise ValueError("bitmap length mismatch")
    writer = BitWriter()
    for bit in bitmap:
        writer.write(bit, 1)
    payload = writer.finish()
    return u32(6 + len(payload)) + u8(6) + u8(0) + payload


def section6_reuse() -> bytes:
    return u32(6) + u8(6) + u8(254)


def section7(
    *,
    initials: Sequence[int],
    minimum_difference: int,
    references: Sequence[int],
    widths: Sequence[int],
    scaled_lengths: Sequence[int],
    groups: Sequence[Sequence[object]],
    options: Section5Options,
    bad_padding: bool = False,
) -> bytes:
    writer = BitWriter()
    extra_width = options.extra_octets * 8
    for initial in initials:
        writer.write(initial, extra_width)
    writer.write(sign_magnitude(minimum_difference, options.extra_octets), extra_width)

    for reference in references:
        writer.write(reference, options.bits_per_value)
    writer.align()

    for width in widths:
        writer.write(width - options.group_width_reference, options.group_width_bits)
    writer.align()

    for scaled_length in scaled_lengths:
        writer.write(scaled_length, options.group_length_bits)
    writer.align()

    # Packed group values are deliberately not octet aligned between groups.
    for group_index, group in enumerate(groups):
        width = widths[group_index]
        for item in group:
            if isinstance(item, str):
                if item == "primary":
                    item = (1 << width) - 1
                elif item == "secondary":
                    item = (1 << width) - 2
                else:
                    raise ValueError(item)
            writer.write(int(item), width)

    payload = writer.finish()
    if bad_padding and payload:
        payload = bytearray(payload)
        payload[-1] |= 1
        payload = bytes(payload)
    return u32(5 + len(payload)) + u8(7) + payload


def build_field(
    *,
    n: int,
    bitmap: Optional[Sequence[int]],
    options: Section5Options,
    initials: Sequence[int],
    minimum_difference: int,
    references: Sequence[int],
    widths: Sequence[int],
    scaled_lengths: Sequence[int],
    groups: Sequence[Sequence[object]],
) -> bytes:
    return (
        section5(options)
        + section6(bitmap, n)
        + section7(
            initials=initials,
            minimum_difference=minimum_difference,
            references=references,
            widths=widths,
            scaled_lengths=scaled_lengths,
            groups=groups,
            options=options,
        )
    )
