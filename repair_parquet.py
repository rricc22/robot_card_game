#!/usr/bin/env python3
"""
Repair parquet files that are missing their footer (FileMetaData).

This happens when LeRobot's ParquetWriter was interrupted before finalize() was called.
The data is intact on disk, only the footer (table of contents) is missing.

Strategy:
1. Read the footer from a working reference file with the same schema
2. Scan the broken file's raw bytes, parsing Thrift-encoded page headers
3. Track column chunk boundaries and sizes
4. Clone and modify the reference footer with correct offsets
5. Append the footer to a copy of the broken file
6. Validate with pyarrow
"""

import copy
import struct
import shutil
import sys
from pathlib import Path

from fastparquet.cencoding import ThriftObject


# === Constants ===
PARQUET_MAGIC = b"PAR1"
PAGE_TYPE_DATA = 0
PAGE_TYPE_DICT = 2
PAGE_TYPE_DATA_V2 = 3


def read_footer(path):
    """Read and parse the FileMetaData footer from a valid parquet file."""
    with open(path, "rb") as f:
        f.seek(-8, 2)
        footer_len = struct.unpack("<I", f.read(4))[0]
        magic = f.read(4)
        assert magic == PARQUET_MAGIC, f"Bad trailing magic: {magic}"
        f.seek(-(8 + footer_len), 2)
        footer_data = bytearray(f.read(footer_len))
    return ThriftObject.from_buffer(footer_data, "FileMetaData")


def scan_pages(file_path, num_leaf_columns):
    """
    Scan a footer-less parquet file and find all page headers.

    Returns a list of row groups, where each row group is a list of column chunk dicts.
    """
    with open(file_path, "rb") as f:
        magic = f.read(4)
        assert magic == PARQUET_MAGIC, f"File doesn't start with PAR1: {magic}"
        file_size = f.seek(0, 2)

    with open(file_path, "rb") as f:
        file_data = bytearray(f.read())

    offset = 4  # skip PAR1 magic
    row_groups = []
    current_row_group = []
    current_column = None
    col_index_in_rg = 0
    pages_parsed = 0

    def new_column():
        return {
            "dict_page_offset": None,
            "data_page_offset": None,
            "start_offset": None,
            "total_compressed_size": 0,
            "total_uncompressed_size": 0,
            "num_values": 0,
        }

    def finish_column():
        nonlocal current_column, col_index_in_rg, current_row_group
        if current_column is not None and current_column["start_offset"] is not None:
            current_row_group.append(current_column)
            col_index_in_rg += 1
            if col_index_in_rg >= num_leaf_columns:
                row_groups.append(current_row_group)
                current_row_group = []
                col_index_in_rg = 0
        current_column = None

    while offset < file_size - 8:
        try:
            header_buf = file_data[offset : offset + 500]
            if len(header_buf) < 10:
                break
            ph = ThriftObject.from_buffer(header_buf, "PageHeader")
            page_type = ph.type
            compressed_size = ph.compressed_page_size
            uncompressed_size = ph.uncompressed_page_size
            header_size = len(ph.to_bytes())

            if compressed_size < 0 or compressed_size > file_size:
                break
            if offset + header_size + compressed_size > file_size:
                break
        except Exception:
            break

        if page_type == PAGE_TYPE_DICT:
            # Dictionary page = start of a new column chunk
            finish_column()
            current_column = new_column()
            current_column["dict_page_offset"] = offset
            current_column["start_offset"] = offset
            current_column["total_compressed_size"] = header_size + compressed_size
            current_column["total_uncompressed_size"] = header_size + uncompressed_size

        elif page_type in (PAGE_TYPE_DATA, PAGE_TYPE_DATA_V2):
            if current_column is None:
                current_column = new_column()
                current_column["start_offset"] = offset
            if current_column["data_page_offset"] is None:
                current_column["data_page_offset"] = offset
            current_column["total_compressed_size"] += header_size + compressed_size
            current_column["total_uncompressed_size"] += header_size + uncompressed_size

            num_values = 0
            if page_type == PAGE_TYPE_DATA and ph.data_page_header is not None:
                num_values = ph.data_page_header.num_values
            elif page_type == PAGE_TYPE_DATA_V2 and ph.data_page_header_v2 is not None:
                num_values = ph.data_page_header_v2.num_values
            current_column["num_values"] += num_values

        offset += header_size + compressed_size
        pages_parsed += 1

    # Finalize last column
    finish_column()

    # Handle incomplete last row group
    if current_row_group:
        if col_index_in_rg == num_leaf_columns:
            row_groups.append(current_row_group)
        else:
            print(f"  WARNING: Last row group has {col_index_in_rg}/{num_leaf_columns} columns "
                  f"— incomplete, discarding")

    print(f"  Scanned {pages_parsed} pages, found {len(row_groups)} row groups, "
          f"data ends at offset {offset}")
    return row_groups, offset


def build_footer_from_ref(ref_footer, scanned_row_groups, num_leaf_columns):
    """
    Build a new FileMetaData by cloning the reference footer and
    replacing row groups with actual scanned data.
    """
    fmd = copy.deepcopy(ref_footer)

    # Build row groups by cloning the reference's first row group as template
    ref_rg_template = ref_footer.row_groups[0]
    new_row_groups = []
    total_rows = 0

    for rg_idx, rg_columns in enumerate(scanned_row_groups):
        rg = copy.deepcopy(ref_rg_template)

        # Determine rows from a scalar column (not a list column)
        # Columns 2+ are scalar (timestamp, frame_index, etc.)
        if len(rg_columns) > 4:
            rg_num_rows = rg_columns[4]["num_values"]  # episode_index
        elif len(rg_columns) > 2:
            rg_num_rows = rg_columns[2]["num_values"]  # timestamp
        else:
            rg_num_rows = rg_columns[0]["num_values"]

        rg.num_rows = rg_num_rows
        total_rows += rg_num_rows

        rg_total_bytes = 0
        for col_idx in range(min(len(rg_columns), len(rg.columns))):
            col_info = rg_columns[col_idx]
            cm = rg.columns[col_idx].meta_data

            cm.num_values = col_info["num_values"]
            cm.total_compressed_size = col_info["total_compressed_size"]
            cm.total_uncompressed_size = col_info["total_uncompressed_size"]
            cm.data_page_offset = col_info["data_page_offset"]
            cm.dictionary_page_offset = col_info["dict_page_offset"]

            # Clear statistics since they won't be correct
            cm.statistics = None

            rg.columns[col_idx].file_offset = (
                col_info["data_page_offset"] or col_info["start_offset"]
            )
            rg_total_bytes += col_info["total_uncompressed_size"]

        rg.total_byte_size = rg_total_bytes
        new_row_groups.append(rg)

    fmd.row_groups = new_row_groups
    fmd.num_rows = total_rows
    return fmd, total_rows


def repair_file(broken_path, output_path, ref_footer, num_leaf_columns, label=""):
    """Repair a single parquet file."""
    print(f"\n  Repairing {label}: {broken_path}")

    scanned_rgs, data_end = scan_pages(broken_path, num_leaf_columns)
    if not scanned_rgs:
        print("  ERROR: No row groups found!")
        return False

    fmd, total_rows = build_footer_from_ref(ref_footer, scanned_rgs, num_leaf_columns)
    print(f"  {len(scanned_rgs)} row groups, {total_rows} total rows")

    footer_bytes = bytes(fmd.to_bytes())
    footer_len = len(footer_bytes)

    # Write: original data up to data_end + footer + footer_len + PAR1
    with open(broken_path, "rb") as f:
        original_data = f.read(data_end)

    with open(output_path, "wb") as f:
        f.write(original_data)
        f.write(footer_bytes)
        f.write(struct.pack("<I", footer_len))
        f.write(PARQUET_MAGIC)

    # Validate with pyarrow
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(output_path))
        meta = pf.metadata
        print(f"  VALID: {meta.num_row_groups} row groups, {meta.num_rows} rows, "
              f"{meta.num_columns} columns")
        table = pf.read_row_group(0)
        print(f"  Read test OK: {table.num_rows} rows from row group 0, "
              f"cols={table.column_names}")
        return True
    except Exception as e:
        print(f"  VALIDATION FAILED: {e}")
        return False


def main():
    base = Path.home() / ".cache/huggingface/lerobot/rricc22"
    output_dir = Path.home() / "lerobot_repaired"
    output_dir.mkdir(exist_ok=True)

    # Reference dataset (working, same schema)
    ref_data_path = base / "robot_card_game_arm_a_win_card_V2/data/chunk-000/file-000.parquet"
    ref_ep_path = base / "robot_card_game_arm_a_win_card_V2/meta/episodes/chunk-000/file-000.parquet"

    print("Loading reference footers...")
    ref_data_footer = read_footer(ref_data_path)
    ref_ep_footer = read_footer(ref_ep_path)
    num_data_cols = len(ref_data_footer.row_groups[0].columns)
    num_ep_cols = len(ref_ep_footer.row_groups[0].columns)
    print(f"  Data: {num_data_cols} leaf columns, Episodes: {num_ep_cols} leaf columns")

    datasets = [
        ("robot_card_game_arm_win_V4", 17, 15171),
        ("robot_card_game_arm_a_win_card_V3", 11, 9856),
    ]

    all_ok = True
    for name, expected_eps, expected_frames in datasets:
        ds_dir = base / name
        out_dir = output_dir / name

        (out_dir / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (out_dir / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Dataset: {name} (expected: {expected_eps} episodes, {expected_frames} frames)")
        print(f"{'='*60}")

        data_ok = repair_file(
            ds_dir / "data/chunk-000/file-000.parquet",
            out_dir / "data/chunk-000/file-000.parquet",
            ref_data_footer,
            num_data_cols,
            "data file",
        )

        ep_ok = repair_file(
            ds_dir / "meta/episodes/chunk-000/file-000.parquet",
            out_dir / "meta/episodes/chunk-000/file-000.parquet",
            ref_ep_footer,
            num_ep_cols,
            "episodes file",
        )

        if data_ok and ep_ok:
            print(f"\n  Dataset {name}: FULLY REPAIRED")
        else:
            print(f"\n  Dataset {name}: data={'OK' if data_ok else 'FAIL'}, "
                  f"episodes={'OK' if ep_ok else 'FAIL'}")
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL FILES REPAIRED SUCCESSFULLY!")
        print(f"Repaired files in: {output_dir}")
        print("\nTo install repaired files:")
        for name, _, _ in datasets:
            s = output_dir / name
            d = base / name
            print(f"  cp {s}/data/chunk-000/file-000.parquet {d}/data/chunk-000/")
            print(f"  cp {s}/meta/episodes/chunk-000/file-000.parquet {d}/meta/episodes/chunk-000/")
    else:
        print("SOME FILES FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
