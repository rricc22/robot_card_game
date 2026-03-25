# Integrating Trained Models into the Card Game

After training on Colab and pushing to HuggingFace, three files need to be updated before running the full game.

---

## Step 1: Note Your Trained Model Repo IDs

At the end of Colab training you pushed two models. They should be at:

```
rricc22/arm_a_play_card_policy
rricc22/arm_b_play_card_policy
```

Confirm they exist at `https://huggingface.co/rricc22` before continuing.

---

## Step 2: Update `config.py`

Open `config.py` and fill in the four values that are currently `None` or placeholders:

```python
# ── Robot hardware ─────────────────────────────────────────────────────────────
ROBOT_PORT = "/dev/ttyACM0"    # ← confirm this matches your validated port from Step 1 of skill1_play_card_commands.md

# ── Cameras ───────────────────────────────────────────────────────────────────
WRIST_CAMERA_INDEX    = "/dev/video4"   # gripper cam — already set
OVERHEAD_CAMERA_INDEX = "/dev/video2"   # ← set to your actual overhead device

# ── LeRobot skill configuration ───────────────────────────────────────────────
PLAY_CARD_EPISODE_TIME_S        = 12    # ← use the value you measured in the dry-run
COLLECT_WINNINGS_EPISODE_TIME_S = 12    # ← measure separately when training collect_winnings

# ── Model paths ────────────────────────────────────────────────────────────────
PLAY_CARD_MODEL_PATH        = "rricc22/arm_a_play_card_policy"
COLLECT_WINNINGS_MODEL_PATH = "rricc22/arm_a_play_card_policy"   # ← update once collect_winnings is trained
```

> As soon as `PLAY_CARD_MODEL_PATH` and `COLLECT_WINNINGS_MODEL_PATH` are both non-`None`, `main.py` automatically switches from `MockRobotInterface` to `LeRobotInterface`. No code changes needed.

---

## Step 3: Smoke Test with Mock First

Before running on hardware, verify the game loop works end-to-end with a forced mock:

```bash
conda activate lerobot_new

# Full mock — no robot, no camera needed
python main.py --mock --no-vision

# Mock robot, real camera (verify vision system reads cards correctly)
python main.py --mock
```

Both should complete a full game without errors. Fix any issues here before touching the robot.

---

## Step 4: Run on Real Hardware

```bash
conda activate lerobot_new

# Permissions (required each reboot unless udev rule is set)
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/video4
sudo chmod 666 /dev/video2   # replace with your overhead device

# Launch the game
python main.py
```

`main.py` will:
1. Detect models are configured → use `LeRobotInterface`
2. Each round: call `lerobot-record --policy.path=rricc22/arm_a_play_card_policy` as a subprocess
3. Wait for the motion to complete
4. Read the cards with the overhead camera
5. Decide winner → trigger `collect_winnings` or `go_home`

---

## Step 5: Tuning If the Robot Misses

| Symptom | Fix in `config.py` |
|---------|-------------------|
| Arm stops before reaching center | Increase `PLAY_CARD_EPISODE_TIME_S` |
| Arm times out waiting after play | Increase `WAIT_AFTER_PLAY` |
| Camera can't read cards consistently | Increase `CARD_READ_ATTEMPTS`; check lighting |
| Motion is jerky / inconsistent | Collect more demos and retrain on Colab |

---

## What Each File Does

| File | Role |
|------|------|
| `config.py` | **Single place to update** — model paths, ports, camera devices, timings |
| `robot_interface.py` | Translates `play_card()` / `collect_winnings()` calls into `lerobot-record` subprocess commands using values from `config.py` |
| `main.py` | Game loop — auto-selects `LeRobotInterface` vs `MockRobotInterface` based on whether model paths are set |
| `vision.py` | Reads card ranks from the overhead camera |
| `game_logic.py` | Pure Python — no robot dependency, compares cards and tracks score |
