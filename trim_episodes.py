#!/usr/bin/env python3
"""
Adaptive episode trimming for LeRobot datasets.

For each episode, detects when the arm stops moving significantly
(task complete) and trims everything after that point.

Strategy: find the last frame where joint velocity exceeds a threshold,
then cut everything after that + a small buffer of frames.
"""

import json
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path


# === CONFIG ===
DATASET_NAME = "robot_card_game_arm_a_play_cardV3"
BASE = Path.home() / ".cache/huggingface/lerobot/rricc22"
OUTPUT = Path.home() / "lerobot_trimmed" / DATASET_NAME

# Trim config
VELOCITY_THRESHOLD = 2.0   # degrees/frame — below this = "not moving"
MIN_STILL_FRAMES = 10      # how many consecutive still frames = "task done"
BUFFER_FRAMES = 5          # keep this many frames after "done" point
MIN_EPISODE_FRAMES = 30    # don't trim below this many frames


def compute_velocity(actions):
    """Compute per-frame joint velocity (absolute change between frames)."""
    vel = np.abs(np.diff(actions, axis=0))
    # Pad with zeros at start to keep same length
    vel = np.vstack([np.zeros((1, vel.shape[1])), vel])
    return vel


def find_trim_point(actions, threshold, min_still, buffer):
    """
    Find the frame index where the task is done.

    Looks for the last moment of significant movement,
    then adds a buffer of frames after it.
    """
    vel = compute_velocity(actions)
    max_vel = vel.max(axis=1)  # max velocity across all joints per frame

    # Find frames where arm is moving
    moving = max_vel > threshold

    # Find the last frame with significant movement
    moving_indices = np.where(moving)[0]
    if len(moving_indices) == 0:
        return len(actions)  # nothing to trim

    last_moving = moving_indices[-1]

    # Check if there are MIN_STILL_FRAMES consecutive still frames after last_moving
    # If not, the arm might just be pausing mid-task
    trim_point = min(last_moving + buffer, len(actions))

    # Verify: after trim_point, the arm should be mostly still
    # If the arm moves again after our trim point, don't trim there
    if trim_point < len(actions) - min_still:
        remaining_vel = max_vel[trim_point:]
        if remaining_vel.max() > threshold * 2:
            # Significant movement after our trim point — find a better one
            # Use the absolute last movement instead
            trim_point = min(moving_indices[-1] + buffer, len(actions))

    return int(trim_point)


def trim_dataset(dataset_name, base_dir, output_dir, threshold=VELOCITY_THRESHOLD,
                 min_still=MIN_STILL_FRAMES, buffer=BUFFER_FRAMES):
    """Trim all episodes in a dataset adaptively."""
    print(f"\n{'='*60}")
    print(f"Trimming: {dataset_name}")
    print(f"{'='*60}")

    ds_dir = base_dir / dataset_name
    data_path = ds_dir / "data/chunk-000/file-000.parquet"
    ep_path = ds_dir / "meta/episodes/chunk-000/file-000.parquet"

    # Read data
    data = pq.read_table(str(data_path))
    eps_meta = pq.read_table(str(ep_path))

    with open(ds_dir / "meta/info.json") as f:
        info = json.load(f)

    actions = np.array([a.as_py() for a in data.column("action")])
    ep_indices = data.column("episode_index").to_pylist()
    unique_eps = sorted(set(ep_indices))

    print(f"Episodes: {len(unique_eps)}, Total frames: {len(actions)}")
    print(f"Threshold: {threshold} deg/frame, Buffer: {buffer} frames")

    # Process each episode
    kept_rows = []
    trim_stats = []

    for ep in unique_eps:
        mask = [i for i, e in enumerate(ep_indices) if e == ep]
        ep_actions = actions[mask]
        original_len = len(ep_actions)

        trim_point = find_trim_point(ep_actions, threshold, min_still, buffer)
        trim_point = max(trim_point, MIN_EPISODE_FRAMES)
        trim_point = min(trim_point, original_len)

        kept = mask[:trim_point]
        kept_rows.extend(kept)

        trimmed = original_len - trim_point
        trim_stats.append((ep, original_len, trim_point, trimmed))
        print(f"  ep{ep:02d}: {original_len} → {trim_point} frames (trimmed {trimmed})")

    print(f"\nTotal frames: {len(actions)} → {len(kept_rows)}")

    # Build trimmed table
    trimmed_data = data.take(kept_rows)

    # Re-index: episode_index stays same, but global index needs recomputing
    new_indices = list(range(len(kept_rows)))
    trimmed_data = trimmed_data.set_column(
        trimmed_data.schema.get_field_index("index"), "index",
        pa.array(new_indices, type=pa.int64())
    )

    # Re-index frame_index within each episode
    new_frame_indices = []
    ep_counter = {}
    for row_ep in trimmed_data.column("episode_index").to_pylist():
        if row_ep not in ep_counter:
            ep_counter[row_ep] = 0
        new_frame_indices.append(ep_counter[row_ep])
        ep_counter[row_ep] += 1

    trimmed_data = trimmed_data.set_column(
        trimmed_data.schema.get_field_index("frame_index"), "frame_index",
        pa.array(new_frame_indices, type=pa.int64())
    )

    # Update episodes metadata
    new_ep_rows = []
    cumulative = 0
    for ep, orig_len, trim_point, _ in trim_stats:
        ep_row_mask = pc.equal(eps_meta.column("episode_index"), ep)
        ep_row = eps_meta.filter(ep_row_mask)
        if ep_row.num_rows == 0:
            continue

        # Update length and indices
        ep_dict = {col: ep_row.column(col).to_pylist()[0] for col in eps_meta.schema.names}
        ep_dict["length"] = trim_point
        ep_dict["dataset_from_index"] = cumulative
        ep_dict["dataset_to_index"] = cumulative + trim_point

        # Update timestamps
        ep_mask = [i for i, e in enumerate(ep_indices) if e == ep]
        timestamps = data.column("timestamp").to_pylist()
        ep_timestamps = [timestamps[i] for i in ep_mask[:trim_point]]
        if ep_timestamps:
            for cam in ["observation.images.wrist", "observation.images.overhead"]:
                if f"videos/{cam}/from_timestamp" in ep_dict:
                    ep_dict[f"videos/{cam}/to_timestamp"] = float(ep_timestamps[-1])

        new_ep_rows.append(ep_dict)
        cumulative += trim_point

    # Rebuild episodes table
    arrays = {}
    for field in eps_meta.schema:
        arrays[field.name] = [row[field.name] for row in new_ep_rows]
    new_eps = pa.table(arrays, schema=eps_meta.schema)

    # Update info.json
    new_info = dict(info)
    new_info["total_frames"] = len(kept_rows)

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

    # Copy tasks and stats
    shutil.copy2(ds_dir / "meta/tasks.parquet", output_dir / "meta/tasks.parquet")
    shutil.copy2(ds_dir / "meta/stats.json", output_dir / "meta/stats.json")

    # Symlink videos (no need to re-encode)
    for cam in ["observation.images.wrist", "observation.images.overhead"]:
        src = ds_dir / f"videos/{cam}/chunk-000"
        dst = output_dir / f"videos/{cam}/chunk-000"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve())

    print(f"\nOutput: {output_dir}")
    print(f"Total frames after trim: {new_info['total_frames']}")
    return output_dir


def preview_trims(dataset_name, base_dir, threshold=VELOCITY_THRESHOLD,
                  min_still=MIN_STILL_FRAMES, buffer=BUFFER_FRAMES):
    """Preview what would be trimmed without writing anything."""
    ds_dir = base_dir / dataset_name
    data_path = ds_dir / "data/chunk-000/file-000.parquet"

    data = pq.read_table(str(data_path))
    actions = np.array([a.as_py() for a in data.column("action")])
    ep_indices = data.column("episode_index").to_pylist()
    unique_eps = sorted(set(ep_indices))

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_roll", "gripper"]

    print(f"\nPREVIEW — threshold={threshold}, buffer={buffer}")
    print(f"{'ep':>4} {'orig':>6} {'trim':>6} {'cut':>6}  last_joint_movement")
    total_orig = 0
    total_trim = 0
    for ep in unique_eps:
        mask = [i for i, e in enumerate(ep_indices) if e == ep]
        ep_actions = actions[mask]
        trim_point = find_trim_point(ep_actions, threshold, min_still, buffer)
        trim_point = max(trim_point, MIN_EPISODE_FRAMES)
        trim_point = min(trim_point, len(ep_actions))

        vel = compute_velocity(ep_actions)
        max_vel = vel.max(axis=1)
        last_move = np.where(max_vel > threshold)[0]
        last_joint = names[vel[last_move[-1]].argmax()] if len(last_move) else "none"

        print(f"{ep:>4} {len(ep_actions):>6} {trim_point:>6} {len(ep_actions)-trim_point:>6}  {last_joint}")
        total_orig += len(ep_actions)
        total_trim += trim_point

    print(f"\nTotal: {total_orig} → {total_trim} ({total_orig - total_trim} frames removed, "
          f"{100*(total_orig-total_trim)/total_orig:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--threshold", type=float, default=VELOCITY_THRESHOLD,
                        help="Min joint velocity (deg/frame) to count as moving")
    parser.add_argument("--buffer", type=int, default=BUFFER_FRAMES,
                        help="Frames to keep after last movement")
    parser.add_argument("--preview", action="store_true",
                        help="Preview trims without writing")
    parser.add_argument("--apply", action="store_true",
                        help="Apply trim and write output")
    args = parser.parse_args()

    if args.preview or not args.apply:
        preview_trims(args.dataset, BASE, threshold=args.threshold, buffer=args.buffer)

    if args.apply:
        out = trim_dataset(args.dataset, BASE, OUTPUT,
                           threshold=args.threshold, buffer=args.buffer)
        print(f"\nDone. Trimmed dataset at: {out}")
        print("To push to HF, update push_datasets.py to include this dataset.")
