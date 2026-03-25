#!/usr/bin/env python3
"""
Trim robot_card_game_arm_win_merged dataset using manual per-episode
timestamps from trim_robot_card_game_win_merged.md.

Source: ~/lerobot_merged/robot_card_game_arm_win_merged/
Output: ~/lerobot_trimmed/robot_card_game_arm_win_merged/

The merged dataset has two data parquet files:
  file-000: EP0-EP10  (V3 data)
  file-001: EP11-EP27 (V4 data)
The output maintains the same two-file structure.
"""

import json
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path


SOURCE_DIR = Path.home() / "lerobot_merged/robot_card_game_arm_win_merged"
OUTPUT_DIR = Path.home() / "lerobot_trimmed/robot_card_game_arm_win_merged"

# EP0-EP10 go in file-000, EP11-EP27 go in file-001
FILE_SPLIT = 11  # first episode index that belongs to file-001

# Per-episode trim points in seconds (keep 0 → trim_s)
TRIM_POINTS = {
    0:  30,
    1:  30,
    2:  30,
    3:  30,
    4:  23,
    5:  25,
    6:  24,
    7:  25,
    8:  21,
    9:  22,
    10: 22,
    11: 27,
    12: 20,
    13: 20,
    14: 23,
    15: 18,
    16: 24,
    17: 21,
    18: 25,
    19: 21,
    20: 21,
    21: 21,
    22: 22,
    23: 19,
    24: 24,
    25: 17,
    26: 22,
    27: 25,
}


def main():
    print(f"\n{'='*60}")
    print("Trimming: robot_card_game_arm_win_merged")
    print(f"{'='*60}")

    # Read both data files
    d0 = pq.read_table(str(SOURCE_DIR / "data/chunk-000/file-000.parquet"))
    d1 = pq.read_table(str(SOURCE_DIR / "data/chunk-000/file-001.parquet"))
    # Concatenate for unified processing
    data = pa.concat_tables([d0, d1])

    eps_meta = pq.read_table(str(SOURCE_DIR / "meta/episodes/chunk-000/file-000.parquet"))

    with open(SOURCE_DIR / "meta/info.json") as f:
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

        if ep in TRIM_POINTS:
            keep_frames = int(TRIM_POINTS[ep] * fps)
            keep_frames = min(keep_frames, orig_frames)
        else:
            keep_frames = orig_frames

        kept = mask[:keep_frames]
        kept_rows.extend(kept)
        ep_trim_info.append((ep, orig_frames, keep_frames))
        print(f"  ep{ep:02d}: {orig_frames} → {keep_frames} frames "
              f"(trimmed {orig_frames - keep_frames}, {TRIM_POINTS.get(ep, '?')}s)")

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
    trimmed_ep_list = trimmed_data.column("episode_index").to_pylist()

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
        trimmed_ep_mask = [i for i, e in enumerate(trimmed_ep_list) if e == ep]
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
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "data/chunk-000").mkdir(parents=True)
    (OUTPUT_DIR / "meta/episodes/chunk-000").mkdir(parents=True)

    # Write data maintaining two-file split: EP0-10 → file-000, EP11-27 → file-001
    for fname, file_eps in [("file-000.parquet", range(0, FILE_SPLIT)),
                             ("file-001.parquet", range(FILE_SPLIT, 28))]:
        writer = pq.ParquetWriter(
            str(OUTPUT_DIR / f"data/chunk-000/{fname}"),
            trimmed_data.schema, compression="snappy"
        )
        for ep in file_eps:
            ep_mask = pc.equal(trimmed_data.column("episode_index"), ep)
            writer.write_table(trimmed_data.filter(ep_mask))
        writer.close()

    pq.write_table(new_eps,
                   str(OUTPUT_DIR / "meta/episodes/chunk-000/file-000.parquet"),
                   compression="snappy")

    with open(OUTPUT_DIR / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    shutil.copy2(SOURCE_DIR / "meta/tasks.parquet", OUTPUT_DIR / "meta/tasks.parquet")
    shutil.copy2(SOURCE_DIR / "meta/stats.json", OUTPUT_DIR / "meta/stats.json")

    # Symlink videos (no re-encoding needed)
    for cam in ["observation.images.wrist", "observation.images.overhead"]:
        src = (SOURCE_DIR / f"videos/{cam}/chunk-000").resolve()
        dst = OUTPUT_DIR / f"videos/{cam}/chunk-000"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)

    # Validate
    check0 = pq.read_table(str(OUTPUT_DIR / "data/chunk-000/file-000.parquet"))
    check1 = pq.read_table(str(OUTPUT_DIR / "data/chunk-000/file-001.parquet"))
    print(f"\nValidation:")
    print(f"  file-000: {check0.num_rows} rows, eps {sorted(set(check0.column('episode_index').to_pylist()))}")
    print(f"  file-001: {check1.num_rows} rows, eps {sorted(set(check1.column('episode_index').to_pylist()))}")
    print(f"  Total: {check0.num_rows + check1.num_rows} rows")
    print(f"\nOutput: {OUTPUT_DIR}")

    # Push to HF
    push = input("\nPush to HF as rricc22/robot_card_game_arm_win_merged? [y/N] ").strip().lower()
    if push == "y":
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = "rricc22/robot_card_game_arm_win_merged"
        print(f"\nPushing to {repo_id}...")
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(OUTPUT_DIR),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["images/**", "*.png"],
        )
        print(f"Done: https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"\nSkipped push. To push manually:")
        print(f"  huggingface-cli upload-large-folder {OUTPUT_DIR} rricc22/robot_card_game_arm_win_merged --repo-type=dataset")


if __name__ == "__main__":
    main()
