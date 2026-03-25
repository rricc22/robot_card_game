#!/usr/bin/env python3
"""
Regenerate complete episodes metadata parquet from recovered data files.

The episodes metadata was only partially flushed before the crash.
This script reads the repaired data file, computes per-episode statistics,
and writes a complete episodes metadata parquet file.
"""

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def compute_quantiles(arr, quantiles=(0.01, 0.10, 0.50, 0.90, 0.99)):
    """Compute quantiles for an array."""
    return {f"q{int(q*100):02d}": float(np.quantile(arr, q)) for q, _ in zip(quantiles, quantiles)}


def compute_stats(values):
    """Compute min, max, mean, std, count, and quantiles for a 1D array."""
    arr = np.array(values, dtype=np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [len(arr)],
        "q01": [float(np.quantile(arr, 0.01))],
        "q10": [float(np.quantile(arr, 0.10))],
        "q50": [float(np.quantile(arr, 0.50))],
        "q90": [float(np.quantile(arr, 0.90))],
        "q99": [float(np.quantile(arr, 0.99))],
    }


def compute_list_stats(values_2d):
    """Compute per-element stats for a list column (e.g., action with 6 elements)."""
    arr = np.array(values_2d, dtype=np.float64)
    result = {}
    for key in ["min", "max", "mean", "std"]:
        func = getattr(np, key) if key != "std" else np.std
        if key == "min":
            result[key] = np.min(arr, axis=0).tolist()
        elif key == "max":
            result[key] = np.max(arr, axis=0).tolist()
        elif key == "mean":
            result[key] = np.mean(arr, axis=0).tolist()
        elif key == "std":
            result[key] = np.std(arr, axis=0).tolist()
    result["count"] = [len(arr)] * arr.shape[1]
    for q_name, q_val in [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]:
        result[q_name] = np.quantile(arr, q_val, axis=0).tolist()
    return result


def compute_image_stats_placeholder(length):
    """
    Create placeholder image stats. Real image stats require decoding video frames,
    which is expensive. We use the existing stats from finalized episodes as reference.
    """
    # Shape is (480, 640, 3) — stats are per-channel
    # Using reasonable defaults for 8-bit images normalized to [0, 1]
    dummy_channel = [0.0]
    dummy_3ch = [[dummy_channel] * 3]
    return {
        "min": dummy_3ch,
        "max": dummy_3ch,
        "mean": dummy_3ch,
        "std": dummy_3ch,
        "count": [length],
        "q01": dummy_3ch,
        "q10": dummy_3ch,
        "q50": dummy_3ch,
        "q90": dummy_3ch,
        "q99": dummy_3ch,
    }


def regenerate_episodes(dataset_name, base_dir, repaired_dir, output_dir):
    """Regenerate the episodes metadata for a dataset."""
    print(f"\n{'='*60}")
    print(f"Regenerating episodes for: {dataset_name}")
    print(f"{'='*60}")

    ds_dir = base_dir / dataset_name
    repaired_data_path = repaired_dir / dataset_name / "data/chunk-000/file-000.parquet"
    existing_ep_path = repaired_dir / dataset_name / "meta/episodes/chunk-000/file-000.parquet"
    info_path = ds_dir / "meta/info.json"
    tasks_path = ds_dir / "meta/tasks.parquet"

    # Read info
    with open(info_path) as f:
        info = json.load(f)
    fps = info["fps"]
    total_episodes_in_info = info["total_episodes"]
    total_frames_in_info = info["total_frames"]

    # Read tasks
    tasks_table = pq.read_table(str(tasks_path))
    task_names = tasks_table.column("task").to_pylist()
    print(f"  Tasks: {task_names}")

    # Read recovered data
    data_table = pq.read_table(str(repaired_data_path))
    print(f"  Data: {data_table.num_rows} rows")

    # Read existing (partial) episodes metadata for reference schema and image stats
    existing_ep = pq.read_table(str(existing_ep_path))
    existing_schema = existing_ep.schema
    print(f"  Existing episodes: {existing_ep.num_rows}")
    print(f"  Schema: {len(existing_schema)} fields")

    # Group data by episode
    ep_indices = data_table.column("episode_index").to_pylist()
    unique_episodes = sorted(set(ep_indices))
    print(f"  Episodes in data: {unique_episodes}")

    # Decide which episodes to include
    # Only include episodes that info.json knows about (skip partial last episode in V4)
    episodes_to_include = [ep for ep in unique_episodes if ep < total_episodes_in_info]
    print(f"  Episodes to include (per info.json): {episodes_to_include}")

    # Build episode rows
    rows = []
    cumulative_index = 0

    for ep_idx in episodes_to_include:
        # Filter data for this episode
        mask = pa.compute.equal(data_table.column("episode_index"), ep_idx)
        ep_data = data_table.filter(mask)
        length = ep_data.num_rows

        # Check if we have existing metadata for this episode
        existing_row = None
        if ep_idx < existing_ep.num_rows:
            existing_row = existing_ep.slice(ep_idx, 1)

        row = {}
        row["episode_index"] = ep_idx
        row["tasks"] = task_names  # all episodes have the same task
        row["length"] = length
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["dataset_from_index"] = cumulative_index
        row["dataset_to_index"] = cumulative_index + length

        # Video info
        row["videos/observation.images.wrist/chunk_index"] = 0
        row["videos/observation.images.wrist/file_index"] = 0
        row["videos/observation.images.overhead/chunk_index"] = 0
        row["videos/observation.images.overhead/file_index"] = 0

        # Video timestamps — absolute position within the video file
        # Each episode is concatenated in the video, so ep_n starts at
        # sum of all previous episode durations
        timestamps = ep_data.column("timestamp").to_pylist()
        ep_duration = float(timestamps[-1])  # last relative timestamp = episode duration
        abs_from = cumulative_index / fps
        abs_to = abs_from + ep_duration
        row["videos/observation.images.wrist/from_timestamp"] = abs_from
        row["videos/observation.images.wrist/to_timestamp"] = abs_to
        row["videos/observation.images.overhead/from_timestamp"] = abs_from
        row["videos/observation.images.overhead/to_timestamp"] = abs_to

        # Compute statistics for action (list<float>[6])
        action_data = [a.as_py() for a in ep_data.column("action")]
        action_stats = compute_list_stats(action_data)
        for stat_name, values in action_stats.items():
            row[f"stats/action/{stat_name}"] = values

        # Compute statistics for observation.state (list<float>[6])
        obs_data = [o.as_py() for o in ep_data.column("observation.state")]
        obs_stats = compute_list_stats(obs_data)
        for stat_name, values in obs_stats.items():
            row[f"stats/observation.state/{stat_name}"] = values

        # Image stats — use existing if available, otherwise use placeholder
        for cam in ["observation.images.wrist", "observation.images.overhead"]:
            if existing_row is not None:
                for stat_name in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
                    col_name = f"stats/{cam}/{stat_name}"
                    row[col_name] = existing_row.column(col_name)[0].as_py()
            else:
                img_stats = compute_image_stats_placeholder(length)
                for stat_name, values in img_stats.items():
                    row[f"stats/{cam}/{stat_name}"] = values

        # Scalar column stats
        for col_name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            col_values = ep_data.column(col_name).to_pylist()
            stats = compute_stats(col_values)
            for stat_name, values in stats.items():
                row[f"stats/{col_name}/{stat_name}"] = values

        # Meta
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        rows.append(row)
        cumulative_index += length

    # Build the table using the existing schema
    # We need to match the exact schema types
    arrays = {}
    for field in existing_schema:
        col_name = field.name
        values = [row[col_name] for row in rows]
        arrays[col_name] = values

    # Create table from dict, then cast to match existing schema
    new_table = pa.table(arrays, schema=existing_schema)

    # Write
    output_ep_path = output_dir / dataset_name / "meta/episodes/chunk-000/file-000.parquet"
    output_ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, str(output_ep_path), compression="snappy")

    # Validate
    check = pq.read_table(str(output_ep_path))
    print(f"\n  Written: {check.num_rows} episodes")
    print(f"  episode_index: {check.column('episode_index').to_pylist()}")
    print(f"  length: {check.column('length').to_pylist()}")
    print(f"  Total frames: {sum(check.column('length').to_pylist())}")
    return True


def main():
    base = Path.home() / ".cache/huggingface/lerobot/rricc22"
    repaired = Path.home() / "lerobot_repaired"
    output = Path.home() / "lerobot_repaired"  # overwrite in same location

    datasets = [
        "robot_card_game_arm_win_V4",
        "robot_card_game_arm_a_win_card_V3",
    ]

    for name in datasets:
        regenerate_episodes(name, base, repaired, output)

    print(f"\n{'='*60}")
    print("Episodes metadata regenerated!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
