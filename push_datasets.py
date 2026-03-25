#!/usr/bin/env python3
"""
Push repaired V3, V4, and merged datasets to Hugging Face.
"""

import json
import copy
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from huggingface_hub import HfApi


BASE = Path.home() / ".cache/huggingface/lerobot/rricc22"
MERGED_DIR = Path.home() / "lerobot_merged/robot_card_game_arm_win_merged"
HF_USER = "rricc22"


def push_dataset(api, local_dir, repo_name):
    """Push a local LeRobot dataset directory to HF."""
    repo_id = f"{HF_USER}/{repo_name}"
    print(f"\n  Pushing to {repo_id}...")

    # Create or get the repo
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload the dataset files (excluding images/ dir which is temp data)
    files_to_upload = []
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and "/images/" not in str(f):
            rel = f.relative_to(local_dir)
            files_to_upload.append((str(f), str(rel)))

    print(f"  Uploading {len(files_to_upload)} files...")
    for local_path, repo_path in files_to_upload:
        size_mb = Path(local_path).stat().st_size / 1024 / 1024
        print(f"    {repo_path} ({size_mb:.1f} MB)")

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["images/**", "*.png"],
    )
    print(f"  Done: https://huggingface.co/datasets/{repo_id}")
    return repo_id


def create_merged_dataset():
    """Create a merged dataset from V3 and V4."""
    print(f"\n{'='*60}")
    print("Creating merged dataset")
    print(f"{'='*60}")

    v3_dir = BASE / "robot_card_game_arm_a_win_card_V3"
    v4_dir = BASE / "robot_card_game_arm_win_V4"

    # Read both data files
    v3_data = pq.read_table(str(v3_dir / "data/chunk-000/file-000.parquet"))
    v4_data = pq.read_table(str(v4_dir / "data/chunk-000/file-000.parquet"))

    v3_eps = pq.read_table(str(v3_dir / "meta/episodes/chunk-000/file-000.parquet"))
    v4_eps = pq.read_table(str(v4_dir / "meta/episodes/chunk-000/file-000.parquet"))

    with open(v3_dir / "meta/info.json") as f:
        v3_info = json.load(f)
    with open(v4_dir / "meta/info.json") as f:
        v4_info = json.load(f)

    n_v3_eps = v3_info["total_episodes"]   # 11
    n_v3_frames = v3_info["total_frames"]  # 9856
    n_v4_eps = v4_info["total_episodes"]   # 17
    n_v4_frames = v4_info["total_frames"]  # 15171

    print(f"  V3: {n_v3_eps} episodes, {n_v3_frames} frames")
    print(f"  V4: {n_v4_eps} episodes, {n_v4_frames} frames")
    print(f"  Merged: {n_v3_eps + n_v4_eps} episodes, {n_v3_frames + n_v4_frames} frames")

    # --- Merge data ---
    # Re-index V4: episode_index += n_v3_eps, index += n_v3_frames
    v4_episode_idx = pc.add(v4_data.column("episode_index"), n_v3_eps)
    v4_global_idx = pc.add(v4_data.column("index"), n_v3_frames)

    v4_reindexed = v4_data.set_column(
        v4_data.schema.get_field_index("episode_index"), "episode_index", v4_episode_idx
    ).set_column(
        v4_data.schema.get_field_index("index"), "index", v4_global_idx
    )

    merged_data = pa.concat_tables([v3_data, v4_reindexed])
    print(f"  Merged data: {merged_data.num_rows} rows")

    # --- Merge episodes metadata ---
    # V3 episodes: file_index=0, V4 episodes: file_index=1
    # Re-index V4 episodes
    v4_ep_idx = pc.add(v4_eps.column("episode_index"), n_v3_eps)
    v4_from_idx = pc.add(v4_eps.column("dataset_from_index"), n_v3_frames)
    v4_to_idx = pc.add(v4_eps.column("dataset_to_index"), n_v3_frames)

    # Set V4 video file_index to 1 (separate video files)
    n_v4_ep_rows = v4_eps.num_rows
    ones = pa.array([1] * n_v4_ep_rows, type=pa.int64())

    v4_eps_reindexed = v4_eps
    v4_eps_reindexed = v4_eps_reindexed.set_column(
        v4_eps_reindexed.schema.get_field_index("episode_index"), "episode_index", v4_ep_idx
    )
    v4_eps_reindexed = v4_eps_reindexed.set_column(
        v4_eps_reindexed.schema.get_field_index("dataset_from_index"), "dataset_from_index", v4_from_idx
    )
    v4_eps_reindexed = v4_eps_reindexed.set_column(
        v4_eps_reindexed.schema.get_field_index("dataset_to_index"), "dataset_to_index", v4_to_idx
    )
    # V4 data goes in file-001
    v4_eps_reindexed = v4_eps_reindexed.set_column(
        v4_eps_reindexed.schema.get_field_index("data/file_index"), "data/file_index", ones
    )
    # V4 videos go in file-001
    for cam in ["observation.images.wrist", "observation.images.overhead"]:
        col = f"videos/{cam}/file_index"
        v4_eps_reindexed = v4_eps_reindexed.set_column(
            v4_eps_reindexed.schema.get_field_index(col), col, ones
        )
    # meta/episodes file_index = 0 for all (single episodes parquet)

    merged_eps = pa.concat_tables([v3_eps, v4_eps_reindexed])
    print(f"  Merged episodes: {merged_eps.num_rows} rows")

    # --- Create output directory ---
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    (MERGED_DIR / "data/chunk-000").mkdir(parents=True)
    (MERGED_DIR / "meta/episodes/chunk-000").mkdir(parents=True)
    (MERGED_DIR / "videos/observation.images.wrist/chunk-000").mkdir(parents=True)
    (MERGED_DIR / "videos/observation.images.overhead/chunk-000").mkdir(parents=True)

    # Write V3 data as file-000, V4 data as file-001
    v3_data_with_groups = pq.ParquetWriter(
        str(MERGED_DIR / "data/chunk-000/file-000.parquet"),
        v3_data.schema, compression="snappy"
    )
    for ep in range(n_v3_eps):
        mask = pc.equal(v3_data.column("episode_index"), ep)
        v3_data_with_groups.write_table(v3_data.filter(mask))
    v3_data_with_groups.close()

    v4_writer = pq.ParquetWriter(
        str(MERGED_DIR / "data/chunk-000/file-001.parquet"),
        v4_reindexed.schema, compression="snappy"
    )
    for ep in range(n_v3_eps, n_v3_eps + n_v4_eps):
        mask = pc.equal(v4_reindexed.column("episode_index"), ep)
        v4_writer.write_table(v4_reindexed.filter(mask))
    v4_writer.close()

    # Write merged episodes
    pq.write_table(merged_eps, str(MERGED_DIR / "meta/episodes/chunk-000/file-000.parquet"),
                    compression="snappy")

    # Copy videos: V3 as file-000, V4 as file-001
    for cam in ["observation.images.wrist", "observation.images.overhead"]:
        shutil.copy2(
            v3_dir / f"videos/{cam}/chunk-000/file-000.mp4",
            MERGED_DIR / f"videos/{cam}/chunk-000/file-000.mp4"
        )
        shutil.copy2(
            v4_dir / f"videos/{cam}/chunk-000/file-000.mp4",
            MERGED_DIR / f"videos/{cam}/chunk-000/file-001.mp4"
        )

    # Copy and merge tasks
    v3_tasks = pq.read_table(str(v3_dir / "meta/tasks.parquet"))
    v4_tasks = pq.read_table(str(v4_dir / "meta/tasks.parquet"))
    # Combine unique tasks
    all_tasks = set(v3_tasks.column("task").to_pylist() + v4_tasks.column("task").to_pylist())
    tasks_table = pa.table({
        "task_index": pa.array(list(range(len(all_tasks))), type=pa.int64()),
        "task": pa.array(list(all_tasks), type=pa.string()),
    })
    pq.write_table(tasks_table, str(MERGED_DIR / "meta/tasks.parquet"), compression="snappy")
    print(f"  Tasks: {list(all_tasks)}")

    # Create merged info.json
    merged_info = copy.deepcopy(v3_info)
    merged_info["total_episodes"] = n_v3_eps + n_v4_eps
    merged_info["total_frames"] = n_v3_frames + n_v4_frames
    merged_info["total_tasks"] = len(all_tasks)
    merged_info["splits"]["train"] = f"0:{n_v3_eps + n_v4_eps}"

    with open(MERGED_DIR / "meta/info.json", "w") as f:
        json.dump(merged_info, f, indent=4)

    # Compute merged stats
    compute_merged_stats(merged_data, MERGED_DIR)

    # Validate
    print("\n  Validating merged dataset...")
    for fname in ["file-000.parquet", "file-001.parquet"]:
        p = MERGED_DIR / f"data/chunk-000/{fname}"
        t = pq.read_table(str(p))
        print(f"    {fname}: {t.num_rows} rows")

    ep_check = pq.read_table(str(MERGED_DIR / "meta/episodes/chunk-000/file-000.parquet"))
    print(f"    episodes: {ep_check.num_rows}")
    print(f"    episode_indices: {ep_check.column('episode_index').to_pylist()}")

    return MERGED_DIR


def compute_merged_stats(data, output_dir):
    """Compute global stats.json for the merged dataset."""
    stats = {}

    # Action stats (list<float>[6])
    action_data = np.array([a.as_py() for a in data.column("action")], dtype=np.float64)
    stats["action"] = {
        "min": np.min(action_data, axis=0).tolist(),
        "max": np.max(action_data, axis=0).tolist(),
        "mean": np.mean(action_data, axis=0).tolist(),
        "std": np.std(action_data, axis=0).tolist(),
        "count": [len(action_data)] * action_data.shape[1],
        "q01": np.quantile(action_data, 0.01, axis=0).tolist(),
        "q10": np.quantile(action_data, 0.10, axis=0).tolist(),
        "q50": np.quantile(action_data, 0.50, axis=0).tolist(),
        "q90": np.quantile(action_data, 0.90, axis=0).tolist(),
        "q99": np.quantile(action_data, 0.99, axis=0).tolist(),
    }

    # Observation.state stats
    obs_data = np.array([o.as_py() for o in data.column("observation.state")], dtype=np.float64)
    stats["observation.state"] = {
        "min": np.min(obs_data, axis=0).tolist(),
        "max": np.max(obs_data, axis=0).tolist(),
        "mean": np.mean(obs_data, axis=0).tolist(),
        "std": np.std(obs_data, axis=0).tolist(),
        "count": [len(obs_data)] * obs_data.shape[1],
        "q01": np.quantile(obs_data, 0.01, axis=0).tolist(),
        "q10": np.quantile(obs_data, 0.10, axis=0).tolist(),
        "q50": np.quantile(obs_data, 0.50, axis=0).tolist(),
        "q90": np.quantile(obs_data, 0.90, axis=0).tolist(),
        "q99": np.quantile(obs_data, 0.99, axis=0).tolist(),
    }

    # Scalar column stats
    for col_name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        arr = np.array(data.column(col_name).to_pylist(), dtype=np.float64)
        stats[col_name] = {
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

    with open(output_dir / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("  Stats computed")


def main():
    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")

    # 1. Push V3
    print(f"\n{'='*60}")
    print("1. Pushing V3 dataset")
    print(f"{'='*60}")
    v3_dir = BASE / "robot_card_game_arm_a_win_card_V3"
    push_dataset(api, v3_dir, "robot_card_game_arm_a_win_card_V3")

    # 2. Push V4
    print(f"\n{'='*60}")
    print("2. Pushing V4 dataset")
    print(f"{'='*60}")
    v4_dir = BASE / "robot_card_game_arm_win_V4"
    push_dataset(api, v4_dir, "robot_card_game_arm_win_V4")

    # 3. Create and push merged dataset
    merged_dir = create_merged_dataset()

    print(f"\n{'='*60}")
    print("3. Pushing merged dataset")
    print(f"{'='*60}")
    push_dataset(api, merged_dir, "robot_card_game_arm_win_merged")

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"  V3: https://huggingface.co/datasets/{HF_USER}/robot_card_game_arm_a_win_card_V3")
    print(f"  V4: https://huggingface.co/datasets/{HF_USER}/robot_card_game_arm_win_V4")
    print(f"  Merged: https://huggingface.co/datasets/{HF_USER}/robot_card_game_arm_win_merged")


if __name__ == "__main__":
    main()
