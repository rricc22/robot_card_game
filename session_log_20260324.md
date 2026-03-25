# Session Log — March 24, 2026

## What We Did

### 1. Parquet Recovery
- Two LeRobot datasets had broken parquet files (recording interrupted before `finalize()`)
- **Affected:** `robot_card_game_arm_win_V4` (17 eps, 15,171 frames) and `robot_card_game_arm_a_win_card_V3` (11 eps, 9,856 frames)
- **Fix:** Wrote `repair_parquet.py` — scans raw bytes, parses Thrift page headers, reconstructs missing footer by deepcopying a reference footer
- **Also fixed:** Episodes metadata parquet (only 10 eps flushed before crash) via `regenerate_episodes.py`
- Repaired files saved to `~/lerobot_repaired/`
- Backups at `~/lerobot_datasets_backup_20260324/`

### 2. Datasets Pushed to HuggingFace
| Dataset | Episodes | Frames | URL |
|---|---|---|---|
| V3 win card | 11 | 9,856 | `rricc22/robot_card_game_arm_a_win_card_V3` |
| V4 win card | 17 | 15,171 | `rricc22/robot_card_game_arm_win_V4` |
| Merged | 28 | 25,027 | `rricc22/robot_card_game_arm_win_merged` |

- Merged dataset: V3 → `file-000`, V4 → `file-001` (no re-encoding needed)
- Unified task name: **"Pick up the winning card and place it in the win zone"**
- Had to manually create `v3.0` git tag (`api.create_tag(...)`) — `upload_folder` doesn't create it
- Had to fix `stats.json` to include image feature keys (`observation.images.wrist`, `observation.images.overhead`) — copied from working V2 dataset

### 3. Training on Google Colab (T4 GPU)
All three policies trained at 10k steps:

| Policy | Dataset | Repo |
|---|---|---|
| Play card | `robot_card_game_arm_a_play_cardV3` | `rricc22/robot_card_game_arm_a_play_cardV3_10k_V2` |
| Win card merged | `robot_card_game_arm_win_merged` | `rricc22/robot_card_game_arm_win_merged_10k` |

Train command template:
```bash
!lerobot-train \
  --policy.type=act \
  --dataset.repo_id=rricc22/<dataset> \
  --steps=10000 \
  --output_dir=outputs/train/<name> \
  --policy.push_to_hub=true \
  --policy.repo_id=rricc22/<name>
```
**Note:** Always clear Colab cache before training: `!rm -rf /root/.cache/huggingface/lerobot/rricc22/<dataset>`

### 4. Inference Testing
**Command template:**
```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --policy.path=rricc22/<model> \
  --dataset.single_task="<task>" \
  --dataset.push_to_hub=false \
  --dataset.repo_id=rricc22/<eval_name> \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=30 \
  --dataset.num_episodes=5
```

---

## Problems Found & Fixed

### Camera device indices swapped
- After replug, `/dev/video4` became the overhead camera and `/dev/video6` became the wrist camera
- Fix: swap `index_or_path` values in the command
- **To avoid:** use `ls /dev/v4l/by-id/` for persistent device paths

### Calibration was overwritten (critical!)
- Ran `lerobot-calibrate` during session which overwrote `follower.json`
- First recalibration: `shoulder_pan` range was only 55 ticks (barely moved during calibration)
- Second recalibration: `elbow_flex` not fully moved
- **Result:** homing offsets changed → robot went to wrong positions during inference
- **Fix:** Restored original calibration from session start (saved from memory)

**Original calibration (used during training — DO NOT OVERWRITE):**
```json
{
  "shoulder_pan":  {"homing_offset": 8,    "range_min": 1363, "range_max": 3207},
  "shoulder_lift": {"homing_offset": -251, "range_min": 879,  "range_max": 3076},
  "elbow_flex":    {"homing_offset": 5,    "range_min": 989,  "range_max": 3081},
  "wrist_flex":    {"homing_offset": 2,    "range_min": 1061, "range_max": 3222},
  "wrist_roll":    {"homing_offset": 108,  "range_min": 0,    "range_max": 4095},
  "gripper":       {"homing_offset": 448,  "range_min": 1521, "range_max": 2506}
}
```
File: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json`

### CPU inference too slow (6 Hz instead of 30 Hz)
- Local GTX 1060 (sm_61) not supported by PyTorch 2.2+
- Policy runs on CPU → ~6 Hz instead of 30 Hz
- ACT action chunk of 100 steps takes ~17s instead of 3.3s
- **Potential fix:** install older PyTorch (≤2.0) in a separate conda env that supports sm_61

### Episodes end with "return to home" motion
- During recording, the arm was brought back to start position before the episode timer ran out
- The policy learned this return motion too
- **Fix (TODO):** trim episodes using the LeRobot visualizer to identify exact trim points per episode

---

### Video timestamps bug in regenerate_episodes.py (fixed)
- `regenerate_episodes.py` was using relative timestamps (0→30s per episode) for `videos/*/from_timestamp` and `videos/*/to_timestamp`
- This caused ALL episodes to point to the first 30s of the video file
- Fix: use `cumulative_index / fps` as absolute `from_timestamp`, `abs_from + last_relative_ts` as `to_timestamp`
- Also: repaired files in `lerobot_repaired/` must be copied back to cache before running `push_datasets.py`

## TODO / Next Steps

1. **Episode trimming** — use LeRobot visualizer to identify trim points for each episode in `robot_card_game_arm_a_play_cardV3`, then trim and retrain
2. **Fix GPU inference** — try installing PyTorch ≤2.0 in a new conda env for 30 Hz inference
3. **Retrain with more steps** — try 50k steps once trimmed data is ready
4. **Record more diverse data** — more episodes, varied lighting/card positions for better generalization

---

## Key Commands Reference

**Calibrate follower arm:**
```bash
lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower
# Type 'c' to recalibrate — WARNING: this overwrites follower.json!
```

**Check camera feeds:**
```bash
ffplay /dev/video4
ffplay /dev/video6
```

**Check available cameras:**
```bash
ls /dev/video*
ls /dev/v4l/by-id/
```

**Record new demos (teleoperation):**
```bash
lerobot-record \
  --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader \
  --dataset.single_task="<task>" \
  --dataset.push_to_hub=false \
  --dataset.repo_id=rricc22/<name> \
  --dataset.num_episodes=20
```
