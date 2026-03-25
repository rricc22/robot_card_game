# Skill 1: `play_card` — Data Collection Commands

**Goal:** Teleoperate each arm to pick up the top card from the deck, place it in the center arena, and return to Home.
**Target:** ~30–50 successful demonstrations per arm.

---

## Step 0: Activate Environment

```bash
conda activate lerobot_new
```

---

## Step 1: Validate & Set Permissions on the Buses

Plug in both arms (leader + follower) then find which port is which:

```bash
# List all connected serial devices
ls /dev/ttyACM*

# Use LeRobot's built-in port finder to identify each arm interactively
lerobot-find-port
```

> Run `lerobot-find-port` once per arm: it will ask you to unplug/replug the cable so it can identify the exact port.

Once you know the ports, grant read/write permissions (required every reboot, or use the udev rule below):

```bash
# Replace ttyACM0 / ttyACM1 with your actual ports
sudo chmod 666 /dev/ttyACM0   # e.g. follower arm A
sudo chmod 666 /dev/ttyACM1   # e.g. leader  arm A
```

**Permanent fix (optional — run once):** add a udev rule so permissions persist across reboots:

```bash
sudo tee /etc/udev/rules.d/99-so100.rules <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Step 2: Validate the Cameras

Two cameras will be used:

| Camera | Device | Purpose |
|--------|--------|---------|
| Gripper / wrist cam | `/dev/video4` | On-arm view during training |
| Static overhead cam | TBD (e.g. `/dev/video2`) | Top-down arena view for card reading |

```bash
# List all video devices
ls /dev/video*

# Quick sanity check — open a preview for each camera (press q to quit)
ffplay /dev/video4
ffplay /dev/video2   # replace with your actual overhead device
```

Grant camera permissions if needed:

```bash
sudo chmod 666 /dev/video4
sudo chmod 666 /dev/video2   # replace as needed
```

**Permanent fix (optional):**

```bash
sudo usermod -aG video $USER
# Then log out and back in
```

---

## Step 3: Dry-Run Teleoperation — Find Episode Timing

Before recording any data, teleoperate freely (no dataset saved) to measure how long the full `play_card` motion takes. You will use this number to set `--robot.episode_time_s` in the recording steps.

```bash
conda activate lerobot_new

lerobot-teleoperate \
  --robot.type=so100 \
  --robot.follower_arms.main.port=/dev/ttyACM0 \
  --robot.leader_arms.main.port=/dev/ttyACM1 \
  --robot.cameras.wrist.type=opencv \
  --robot.cameras.wrist.index=/dev/video4 \
  --robot.cameras.wrist.width=640 \
  --robot.cameras.wrist.height=480 \
  --robot.cameras.wrist.fps=30 \
  --robot.cameras.overhead.type=opencv \
  --robot.cameras.overhead.index=/dev/video6 \
  --robot.cameras.overhead.width=640 \
  --robot.cameras.overhead.height=480 \
  --robot.cameras.overhead.fps=30 \
  --display_cameras=true
  
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video4, width: 640,
height: 480, fps: 30}, overhead: {type: opencv, index_or_path: /dev/video6, width:
640, height: 480, fps: 30}}" \
    --display_data=true
```

**What to do during the dry-run:**

1. Start a stopwatch when you begin the motion.
2. Execute the full sequence at a comfortable, repeatable pace:
   - Move to the deck → grip card → move to center arena → release → return to **Home**
3. Stop the stopwatch when the arm is back at Home.
4. Repeat 3–5 times and note the **longest** duration (you want a small margin).
5. Round up to the nearest second and add **+2 s** buffer.

**Example:**

| Trial | Time |
|-------|------|
| 1     | 8.2 s |
| 2     | 9.1 s |
| 3     | 8.7 s |
| **Use** | **12 s** (longest ~9 s + 2 s buffer, rounded up) |

> Update `--robot.episode_time_s=12` (or your measured value) in **Steps 4 and 5** below before recording.

Press `Ctrl+C` to exit teleoperation when done.

---

## Step 4: Record `play_card` Demos — Arm A

```bash
conda activate lerobot_new

# ↓ Set this to your measured value from Step 3
EPISODE_TIME=12

lerobot-record \
  --robot.type=so100 \
  --robot.follower_arms.main.port=/dev/ttyACM0 \
  --robot.leader_arms.main.port=/dev/ttyACM1 \
  --robot.cameras.wrist.type=opencv \
  --robot.cameras.wrist.index=/dev/video4 \
  --robot.cameras.wrist.width=640 \
  --robot.cameras.wrist.height=480 \
  --robot.cameras.wrist.fps=30 \
  --robot.cameras.overhead.type=opencv \
  --robot.cameras.overhead.index=/dev/video6 \
  --robot.cameras.overhead.width=640 \
  --robot.cameras.overhead.height=480 \
  --robot.cameras.overhead.fps=30 \
  --robot.episode_time_s=$EPISODE_TIME \
  --dataset.repo_id=rricc22/robot_card_game_arm_a_play_card \
  --dataset.num_episodes=50 \
  --dataset.push_to_hub=true \
  --display_cameras=true
  
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video4, width: 640,
  height: 480, fps: 30}, overhead: {type: opencv, index_or_path: /dev/video6, width:
  640, height: 480, fps: 30}}" \
    --dataset.repo_id=rricc22/robot_card_game_arm_a_play_card \
    --dataset.num_episodes=50 \
    --dataset.push_to_hub=true \
    --episode_time_s=30 \
    --reset_time_s=10 \
    --display_data=true
```

> **During recording — each episode:**
> 1. Reach to the deck
> 2. Grip the top card
> 3. Move smoothly to the center arena
> 4. Release the card
> 5. Return to neutral **Home** position
> 6. Confirm to save the episode, or discard and redo.

---

## Step 5: Record `play_card` Demos — Arm B

> Plug in Arm B's follower + leader and run `lerobot-find-port` again to get its ports (e.g. `/dev/ttyACM2`, `/dev/ttyACM3`). Grant permissions the same way as Step 1.
> Run the dry-run teleoperation from Step 3 again for Arm B if its reach/position differs from Arm A — the timing may be slightly different.

```bash
sudo chmod 666 /dev/ttyACM2
sudo chmod 666 /dev/ttyACM3

conda activate lerobot_new

# ↓ Use same value as Step 4, or re-measure for Arm B if positioning differs
EPISODE_TIME=12

lerobot-record \
  --robot.type=so100 \
  --robot.follower_arms.main.port=/dev/ttyACM2 \
  --robot.leader_arms.main.port=/dev/ttyACM3 \
  --robot.cameras.wrist.type=opencv \
  --robot.cameras.wrist.index=/dev/video4 \
  --robot.cameras.wrist.width=640 \
  --robot.cameras.wrist.height=480 \
  --robot.cameras.wrist.fps=30 \
  --robot.cameras.overhead.type=opencv \
  --robot.cameras.overhead.index=/dev/video2 \
  --robot.cameras.overhead.width=640 \
  --robot.cameras.overhead.height=480 \
  --robot.cameras.overhead.fps=30 \
  --robot.episode_time_s=$EPISODE_TIME \
  --dataset.repo_id=rricc22/robot_card_game_arm_b_play_card \
  --dataset.num_episodes=50 \
  --dataset.push_to_hub=true \
  --display_cameras=true
```

---

## Step 6: Verify Datasets on HuggingFace

After recording, confirm both datasets are pushed:
- `https://huggingface.co/datasets/rricc22/robot_card_game_arm_a_play_card`
- `https://huggingface.co/datasets/rricc22/robot_card_game_arm_b_play_card`

---

## Step 7: Train on Google Colab

> **Do NOT train locally** — the GTX 1060 (sm_61) is not supported by PyTorch 2.2+. Use Google Colab (T4 GPU).

```python
# On Colab — run once to install
!pip install lerobot

# Train Arm A
!lerobot-train \
  --dataset.repo_id=rricc22/robot_card_game_arm_a_play_card \
  --policy.type=act \
  --output_dir=outputs/arm_a_play_card \
  --wandb.enable=false \
  --push_to_hub=true \
  --hub.repo_id=rricc22/arm_a_play_card_policy

# Train Arm B
!lerobot-train \
  --dataset.repo_id=rricc22/robot_card_game_arm_b_play_card \
  --policy.type=act \
  --output_dir=outputs/arm_b_play_card \
  --wandb.enable=false \
  --push_to_hub=true \
  --hub.repo_id=rricc22/arm_b_play_card_policy
```

---

## Step 8: Evaluate / Run Inference (after training)

```bash
conda activate lerobot_new

# Arm A
lerobot-record \
  --robot.type=so100 \
  --robot.follower_arms.main.port=/dev/ttyACM0 \
  --robot.cameras.wrist.type=opencv \
  --robot.cameras.wrist.index=/dev/video4 \
  --robot.cameras.wrist.width=640 \
  --robot.cameras.wrist.height=480 \
  --robot.cameras.wrist.fps=30 \
  --robot.cameras.overhead.type=opencv \
  --robot.cameras.overhead.index=/dev/video2 \
  --robot.cameras.overhead.width=640 \
  --robot.cameras.overhead.height=480 \
  --robot.cameras.overhead.fps=30 \
  --policy.path=rricc22/arm_a_play_card_policy \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=1

# Arm B
lerobot-record \
  --robot.type=so100 \
  --robot.follower_arms.main.port=/dev/ttyACM2 \
  --robot.cameras.wrist.type=opencv \
  --robot.cameras.wrist.index=/dev/video4 \
  --robot.cameras.wrist.width=640 \
  --robot.cameras.wrist.height=480 \
  --robot.cameras.wrist.fps=30 \
  --robot.cameras.overhead.type=opencv \
  --robot.cameras.overhead.index=/dev/video2 \
  --robot.cameras.overhead.width=640 \
  --robot.cameras.overhead.height=480 \
  --robot.cameras.overhead.fps=30 \
  --policy.path=rricc22/arm_b_play_card_policy \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=1
```
