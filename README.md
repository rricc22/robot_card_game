# Robot Card Game

A robot arm that plays a card game using the [LeRobot](https://github.com/huggingface/lerobot) framework and an ACT (Action Chunking with Transformers) policy.

## Hardware

- SO-100 follower arm
- 2 cameras: wrist + overhead
- LeRobot for data collection and training

## Datasets (Hugging Face)

| Dataset | Description |
|---|---|
| [robot_card_game_arm_win_merged](https://huggingface.co/datasets/rricc22/robot_card_game_arm_win_merged) | Full merged dataset (28 episodes) |
| [robot_card_game_arm_win_merged_trimmed](https://huggingface.co/datasets/rricc22/robot_card_game_arm_win_merged_trimmed) | Trimmed version — idle frames removed |
| [robot_card_game_arm_a_play_cardV3](https://huggingface.co/datasets/rricc22/robot_card_game_arm_a_play_cardV3) | Play card skill dataset |
| [robot_card_game_arm_a_play_cardV3_trimmed](https://huggingface.co/datasets/rricc22/robot_card_game_arm_a_play_cardV3_trimmed) | Trimmed play card dataset |

## Trained Policies (Hugging Face)

| Policy | Dataset | Steps |
|---|---|---|
| [robot_card_game_arm_win_merged_trimmed_50k](https://huggingface.co/rricc22/robot_card_game_arm_win_merged_trimmed_50k) | win_merged_trimmed | 50k |

## Project Structure

```
main.py                  # Main game loop
game_logic.py            # Card game rules
vision.py                # Camera + card detection (EasyOCR)
display.py               # Terminal UI
robot_interface.py       # LeRobot arm control
config.py                # Hardware config (ports, cameras)

apply_trim.py            # Trim play_card dataset by episode
apply_trim_win_merged.py # Trim win_merged dataset by episode
trim_episodes.py         # Adaptive velocity-based trimming
push_datasets.py         # Merge V3+V4 and push to HF
repair_parquet.py        # Fix corrupted parquet files
regenerate_episodes.py   # Rebuild episode metadata
```

## Training

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=rricc22/robot_card_game_arm_win_merged_trimmed \
    --steps=50000 \
    --output_dir=outputs/train/robot_card_game_arm_win_merged_trimmed \
    --policy.push_to_hub=true \
    --policy.repo_id=rricc22/robot_card_game_arm_win_merged_trimmed_50k
```

## Running

```bash
pip install -r requirements.txt
python main.py
```
