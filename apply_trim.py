#!/usr/bin/env python3
"""
Apply manual per-episode trim points to a LeRobot dataset.
Trim points are specified in seconds; frames = seconds * fps.
"""

import json
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path


BASE = Path.home() / ".cache/huggingface/lerobot/rricc22"
OUTPUT_BASE = Path.home() / "lerobot_trimmed"


# Per-episode trim points in seconds (keep 0 → trim_s, discard rest)
TRIM_POINTS = {
    "robot_card_game_arm_a_play_cardV3": {
        0:  22,
        1:  24,
        2:  23,
        3:  25,
        4:  24,
        5:  24,
        6:  23,
        7:  23,
        8:  20,
        9:  22,
        10: 23,
        11: 22,
        12: 19,
        13: 22,
        14: 20,
        15: 23,
        16: 21,
        17: 23,
        18: 23,
        19: 24,
        20: 17,
        21: 21,
        22: 21,
        23: 21,
        24: 26,
        25: 24,
        26: 18,
        27: 18,
        28: 24,
        29: 20,
    }
}


def apply_trim(dataset_name, trim_points_s):
    ds_dir = BASE / dataset_name
    output_dir = OUTPUT_BASE / dataset_name

    print(f"\n{'='*60}")
    print(f"Trimming: {dataset_name}")
    print(f"{'='*60}")

    data = pq.read_table(str(ds_dir / "data/chunk-000/file-000.parquet"))
    eps_meta = pq.read_table(str(ds_dir / "meta/episodes/chunk-000/file-000.parquet"))

    with open(ds_dir / "meta/info.json") as f:
        info = json.load(f)

    fps = info["fps"]
    ep_indices = data.column("episode_index").to_pylist()
    unique_eps = sorted(set(ep_indices))

    print(f"FPS: {fps}, Episodes: {len(unique_eps)}, Total frames: {len(ep_indices)}")

    # Build list of row indices to keep
    kept_rows = []
    ep_trim_info = []  # (ep, orig_frames, kept_frames)

    for ep in unique_eps:
        mask = [i for i, e in enumerate(ep_indices) if e == ep]
        orig_frames = len(mask)

        if ep in trim_points_s:
            keep_frames = int(trim_points_s[ep] * fps)
            keep_frames = min(keep_frames, orig_frames)
        else:
            keep_frames = orig_frames
            print(f"  ep{ep:02d}: no trim point specified, keeping all {orig_frames} frames")

        kept = mask[:keep_frames]
        kept_rows.extend(kept)
        ep_trim_info.append((ep, orig_frames, keep_frames))
        print(f"  ep{ep:02d}: {orig_frames} frames → {keep_frames} frames "
              f"(trimmed {orig_frames - keep_frames}, {trim_points_s.get(ep, '?')}s)")

    total_kept = len(kept_rows)
    total_orig = len(ep_indices)
    print(f"\nTotal: {total_orig} → {total_kept} frames "
          f"({total_orig - total_kept} removed, {100*(total_orig-total_kept)/total_orig:.1f}%)")

    # Build trimmed data table
    trimmed_data = data.take(kept_rows)

    # Re-index global index
    trimmed_data = trimmed_data.set_column(
        trimmed_data.schema.get_field_index("index"), "index",
        pa.array(list(range(total_kept)), type=pa.int64())
    )

    # Re-index frame_index within each episode
    new_frame_indices = []
    ep_counter = {}
    for row_ep in trimmed_data.column("episode_index").to_pylist():
        ep_counter[row_ep] = ep_counter.get(row_ep, 0)
        new_frame_indices.append(ep_counter[row_ep])
        ep_counter[row_ep] += 1

    trimmed_data = trimmed_data.set_column(
        trimmed_data.schema.get_field_index("frame_index"), "frame_index",
        pa.array(new_frame_indices, type=pa.int64())
    )

    # Update episodes metadata
    new_ep_rows = []
    cumulative = 0
    for ep, orig_frames, keep_frames in ep_trim_info:
        ep_filter = pc.equal(eps_meta.column("episode_index"), ep)
        ep_row = eps_meta.filter(ep_filter)
        if ep_row.num_rows == 0:
            print(f"  WARNING: no metadata row for ep{ep}, skipping")
            continue

        ep_dict = {col: ep_row.column(col).to_pylist()[0]
                   for col in eps_meta.schema.names}

        ep_dict["length"] = keep_frames
        ep_dict["dataset_from_index"] = cumulative
        ep_dict["dataset_to_index"] = cumulative + keep_frames

        # Update video end timestamp
        trimmed_ep_mask = [i for i, e in
                           enumerate(trimmed_data.column("episode_index").to_pylist())
                           if e == ep]
        if trimmed_ep_mask:
            last_relative_ts = trimmed_data.column("timestamp")[trimmed_ep_mask[-1]].as_py()
            for cam in ["observation.images.wrist", "observation.images.overhead"]:
                from_key = f"videos/{cam}/from_timestamp"
                to_key = f"videos/{cam}/to_timestamp"
                if to_key in ep_dict and from_key in ep_dict:
                    ep_dict[to_key] = float(ep_dict[from_key]) + float(last_relative_ts)

        new_ep_rows.append(ep_dict)
        cumulative += keep_frames

    arrays = {field.name: [row[field.name] for row in new_ep_rows]
              for field in eps_meta.schema}
    new_eps = pa.table(arrays, schema=eps_meta.schema)

    # Update info.json
    new_info = dict(info)
    new_info["total_frames"] = total_kept

    # Write output
    if output_dir.exists():
        shutil.rmtree(output_dir)

    (output_dir / "data/chunk-000").mkdir(parents=True)
    (output_dir / "meta/episodes/chunk-000").mkdir(parents=True)

    # Write data with one row group per episode
    writer = pq.ParquetWriter(
        str(output_dir / "data/chunk-000/file-000.parquet"),
        trimmed_data.schema, compression="snappy"
    )
    for ep in unique_eps:
        ep_mask = pc.equal(trimmed_data.column("episode_index"), ep)
        writer.write_table(trimmed_data.filter(ep_mask))
    writer.close()

    pq.write_table(new_eps,
                   str(output_dir / "meta/episodes/chunk-000/file-000.parquet"),
                   compression="snappy")

    with open(output_dir / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    shutil.copy2(ds_dir / "meta/tasks.parquet",
                 output_dir / "meta/tasks.parquet")
    shutil.copy2(ds_dir / "meta/stats.json",
                 output_dir / "meta/stats.json")

    # Symlink videos (no re-encoding needed)
    for cam in ["observation.images.wrist", "observation.images.overhead"]:
        src = (ds_dir / f"videos/{cam}/chunk-000").resolve()
        dst = output_dir / f"videos/{cam}/chunk-000"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)

    # Validate
    check = pq.read_table(str(output_dir / "data/chunk-000/file-000.parquet"))
    print(f"\nValidation: {check.num_rows} rows, "
          f"{check.column('episode_index').to_pylist()[-1]+1} episodes")
    print(f"Output: {output_dir}")
    return output_dir


if __name__ == "__main__":
    for dataset_name, trim_points in TRIM_POINTS.items():
        apply_trim(dataset_name, trim_points)

    print("\nDone! Next steps:")
    print("1. Push trimmed dataset to HF")
    print("2. Retrain on Colab with trimmed data")
