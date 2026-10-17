"""Gzip TFRecords that GRID's ``TFRecordIterator`` can read, without TensorFlow.

``tiger_inference_flat`` opens ``*.tfrecord.gz`` with
``TFRecordDataset(..., compression_type="GZIP")`` and infers an int64
``VarLenFeature`` from each ``tf.train.Example``. The records this module
writes use that framing and a packed ``int64_list`` for every feature.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

_CRC32C_POLY = 0x82F63B78


def _build_crc32c_table() -> tuple[int, ...]:
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ _CRC32C_POLY
            else:
                crc >>= 1
        table.append(crc)
    return tuple(table)


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(data: bytes) -> int:
    """Castagnoli CRC32C. ``crc32c(b"123456789") == 0xE3069283``."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    """Mask used by the TFRecord framing (uint32 wrap)."""
    crc = crc32c(data)
    rotated = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"TFRecord int64 features must be >= 0, got {value}")
    out = bytearray()
    while True:
        piece = value & 0x7F
        value >>= 7
        if value:
            out.append(piece | 0x80)
        else:
            out.append(piece)
            return bytes(out)


def _encode_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_len_field(field_number: int, payload: bytes) -> bytes:
    return _encode_key(field_number, 2) + _encode_varint(len(payload)) + payload


def encode_example(features: dict[str, list[int]]) -> bytes:
    """Encode a ``tf.train.Example`` whose features are packed int64 lists.

    Map entries are written in sorted feature-name order. Parsers accept any
    order; sorting keeps the bytes stable.
    """
    if not features:
        raise ValueError("an example needs at least one feature")
    entries = []
    for name in sorted(features):
        values = features[name]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"feature {name} must be a list of non-negative ints")
        packed = b"".join(_encode_varint(int(value)) for value in values)
        int64_list = _encode_len_field(1, packed)
        feature = _encode_len_field(3, int64_list)
        entry = _encode_len_field(1, name.encode("utf-8")) + _encode_len_field(2, feature)
        entries.append(_encode_len_field(1, entry))
    return _encode_len_field(1, b"".join(entries))


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("varint is too long")


def _read_len_field(data: bytes, offset: int, expected_field: int) -> tuple[bytes, int]:
    key, offset = _read_varint(data, offset)
    field_number = key >> 3
    wire_type = key & 0x07
    if field_number != expected_field or wire_type != 2:
        raise ValueError(f"expected length-delimited field {expected_field}, got field {field_number} wire {wire_type}")
    length, offset = _read_varint(data, offset)
    end = offset + length
    if end > len(data):
        raise ValueError("truncated length-delimited field")
    return data[offset:end], end


def _decode_packed_int64s(payload: bytes) -> list[int]:
    values = []
    offset = 0
    while offset < len(payload):
        value, offset = _read_varint(payload, offset)
        values.append(value)
    return values


def decode_example(payload: bytes) -> dict[str, list[int]]:
    """Decode an example produced by :func:`encode_example`."""
    features_body, end = _read_len_field(payload, 0, 1)
    if end != len(payload):
        raise ValueError("trailing bytes after Example.features")
    decoded: dict[str, list[int]] = {}
    offset = 0
    while offset < len(features_body):
        entry, offset = _read_len_field(features_body, offset, 1)
        name_bytes, entry_offset = _read_len_field(entry, 0, 1)
        feature, entry_end = _read_len_field(entry, entry_offset, 2)
        if entry_end != len(entry):
            raise ValueError("trailing bytes in a feature map entry")
        int64_list, feature_end = _read_len_field(feature, 0, 3)
        if feature_end != len(feature):
            raise ValueError("feature is not a single int64_list")
        if int64_list:
            packed, packed_end = _read_len_field(int64_list, 0, 1)
            if packed_end != len(int64_list):
                raise ValueError("trailing bytes in int64_list")
            values = _decode_packed_int64s(packed)
        else:
            values = []
        name = name_bytes.decode("utf-8")
        if name in decoded:
            raise ValueError(f"duplicate feature {name}")
        decoded[name] = values
    return decoded


def write_tfrecord_gz(path, records: list[dict[str, list[int]]]) -> None:
    """Write one gzip TFRecord file. ``records`` map feature name to int64s."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wb") as handle:
        for record in records:
            payload = encode_example(record)
            length = struct.pack("<Q", len(payload))
            handle.write(length)
            handle.write(struct.pack("<I", masked_crc32c(length)))
            handle.write(payload)
            handle.write(struct.pack("<I", masked_crc32c(payload)))


def read_tfrecord_gz(path) -> list[dict[str, list[int]]]:
    """Read a file written by :func:`write_tfrecord_gz` and check both CRCs."""
    records = []
    with gzip.open(path, "rb") as handle:
        blob = handle.read()
    offset = 0
    while offset < len(blob):
        if offset + 12 > len(blob):
            raise ValueError(f"{path} ended inside a TFRecord header")
        length = struct.unpack_from("<Q", blob, offset)[0]
        length_crc = struct.unpack_from("<I", blob, offset + 8)[0]
        if length_crc != masked_crc32c(blob[offset : offset + 8]):
            raise ValueError(f"{path} has a bad length CRC at offset {offset}")
        start = offset + 12
        end = start + length
        if end + 4 > len(blob):
            raise ValueError(f"{path} ended inside a TFRecord payload")
        payload = blob[start:end]
        data_crc = struct.unpack_from("<I", blob, end)[0]
        if data_crc != masked_crc32c(payload):
            raise ValueError(f"{path} has a bad data CRC at offset {start}")
        records.append(decode_example(payload))
        offset = end + 4
    return records
