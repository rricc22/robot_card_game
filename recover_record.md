# Parquet Recovery — COMPLETED

## The Problem

The parquet files were incomplete — the recording was interrupted before `finalize()` was called.

When LeRobot records data, its `ParquetWriter`:
1. Writes all the actual data (joint positions, frame indices, etc.) to disk as it records
2. **Never writes the footer** — a small index at the end of the file that tells readers where all the data is

Every parquet reader (pyarrow, duckdb, polars, fastparquet) **requires the footer** to read the file. Without it, they all refuse to open it.

The `finalize()` method is what closes the writer and writes the footer. It should have been called at the end of recording, but wasn't.

---

## Affected Datasets

Located at: `~/.cache/huggingface/lerobot/rricc22/`

| Dataset | Episodes | Frames | Recorded |
|---|---|---|---|
| `robot_card_game_arm_win_V4` | 17 | 15,171 | Mar 24, 16:13 |
| `robot_card_game_arm_a_win_card_V3` | 11 | 9,856 | Mar 24, 15:45 |

Broken files in each dataset:
- `data/chunk-000/file-000.parquet` — frame data (actions, states, indices)
- `meta/episodes/chunk-000/file-000.parquet` — episode metadata (only 10 of N episodes were flushed before crash)

OK files (already had proper footers):
- `meta/tasks.parquet`
- `meta/info.json`
- `meta/stats.json`
- `videos/` — all `.mp4` files

---

## What Was Done (Investigation)

1. **Identified the root cause**: `finalize()` was never called, so `ParquetWriter.close()` never ran → footer never written.
2. **Confirmed backups are saved** at: `~/lerobot_datasets_backup_20260324/`
3. **Confirmed the data IS in the files** (starts with PAR1 magic, data follows) — just no footer.
4. **Tried all readers** — pyarrow, fastparquet, duckdb, polars — all fail without the footer.
5. **Found the schema** from a working dataset (`robot_card_game_arm_a_win_card_V2`):
   - Columns: `action.list.element` (FLOAT), `observation.state.list.element` (FLOAT), `timestamp` (FLOAT), `frame_index` (INT64), `episode_index` (INT64), `index` (INT64), `task_index` (INT64)
   - 7 columns total, each episode = 1 row group, snappy compressed, dictionary encoding

---

## Repair — COMPLETED (Mar 24, 2026)

### Step 1: Footer Reconstruction (`repair_parquet.py`)

Used `fastparquet.cencoding.ThriftObject` to:
1. **Read reference footer** from working V2 dataset (`from_buffer` to parse the FileMetaData)
2. **Scan broken files** page-by-page — parsed Thrift-encoded PageHeaders to locate every dictionary page and data page, tracking column chunk boundaries, offsets, compressed/uncompressed sizes, and num_values
3. **Built new footer** by deep-copying the reference FileMetaData and replacing row group metadata with correct offsets from the scan
4. **Appended footer** to each broken file: `serialized_FileMetaData + 4-byte_length + b"PAR1"`

Results:
- **V4 data**: 18 row groups found (17 complete episodes + 1 partial episode of 339 frames that was being recorded when the laptop shut down). Trimmed to 17 episodes / 15,171 frames to match `info.json`.
- **V3 data**: 11 row groups, 9,856 frames — exact match.
- **V4 episodes metadata**: only 10 of 17 episodes had been flushed.
- **V3 episodes metadata**: only 10 of 11 episodes had been flushed.

### Step 2: Episodes Metadata Regeneration (`regenerate_episodes.py`)

The episodes parquet files only had metadata for the first 10 episodes (the rest weren't flushed before crash). Regenerated complete metadata by:
1. Reading all episode data from the repaired data files
2. Reusing existing per-episode stats (including image stats) for episodes 0-9
3. Computing fresh stats (action, observation.state, timestamp, frame_index, etc.) from the data for episodes 10+
4. Image stats for episodes 10+ use placeholders (computing real ones requires full video decode)

### Step 3: Push to Hugging Face (`push_datasets.py`)

Pushed all three datasets:

| Dataset | Episodes | Frames | URL |
|---|---|---|---|
| V3 | 11 | 9,856 | https://huggingface.co/datasets/rricc22/robot_card_game_arm_a_win_card_V3 |
| V4 | 17 | 15,171 | https://huggingface.co/datasets/rricc22/robot_card_game_arm_win_V4 |
| **Merged** | **28** | **25,027** | https://huggingface.co/datasets/rricc22/robot_card_game_arm_win_merged |

### Step 4: Merged Dataset

Created a combined dataset from V3 + V4 (same task, recording split across two sessions due to laptop shutdown):
- V3 episodes 0-10 → merged episodes 0-10 (data in `chunk-000/file-000.parquet`, videos in `file-000.mp4`)
- V4 episodes 0-16 → merged episodes 11-27 (data in `chunk-000/file-001.parquet`, videos in `file-001.mp4`)
- Unified task name: **"Pick up the winning card and place it in the win zone"**
- Re-indexed `episode_index`, `index`, `dataset_from_index`, `dataset_to_index`
- Computed fresh global `stats.json` across all 25,027 frames

---

## Files Created

| Script | Purpose |
|---|---|
| `repair_parquet.py` | Scans broken parquet files and reconstructs missing footers |
| `regenerate_episodes.py` | Regenerates complete episodes metadata from repaired data |
| `push_datasets.py` | Creates merged dataset and pushes all three to Hugging Face |

Repaired files cached at: `~/lerobot_repaired/`
Merged dataset cached at: `~/lerobot_merged/`
Original backups at: `~/lerobot_datasets_backup_20260324/`

---

## Key Technical Details

- **conda env**: `lerobot_new` (pyarrow 23.0.1, fastparquet 2026.3.0)
- **Thrift parsing**: `fastparquet.cencoding.ThriftObject.from_buffer(data, 'PageHeader')` to read page headers; `.to_bytes()` to serialize footers
- **Footer cloning**: deep-copying reference footer preserves the exact Thrift field structure that pyarrow expects (building from scratch with `from_fields` produces incompatible serialization)
- **Data file schema**: 12 schema elements (7 leaf columns), fixed_size_list<float>[6] for action/observation.state
- **Episodes file schema**: 362 schema elements (107 leaf columns), includes per-episode stats for all data columns + image channels
