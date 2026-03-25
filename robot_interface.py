"""
Robot interface layer.

MockRobotInterface  — runs without any robot (for testing game logic & vision)
LeRobotInterface    — calls `lerobot-record --policy.path=...` via subprocess,
                      exactly like you already use for evaluation.

HOW TO USE LeRobotInterface:
  1. Train play_card and collect_winnings policies (push to HuggingFace or keep local).
  2. Set PLAY_CARD_MODEL_PATH and COLLECT_WINNINGS_MODEL_PATH in config.py.
  3. Run:  conda activate lerobot && python main.py

  The interface calls lerobot-record with --dataset.num_episodes=1 per skill,
  waits for it to finish, then returns. No HuggingFace push — logs stay local.

  Tune PLAY_CARD_EPISODE_TIME_S and COLLECT_WINNINGS_EPISODE_TIME_S in config.py
  to match how long each motion takes in your demos.
"""

import subprocess
import sys
import time
from abc import ABC, abstractmethod

from config import (
    COLLECT_WINNINGS_EPISODE_TIME_S,
    COLLECT_WINNINGS_MODEL_PATH,
    GAME_LOG_REPO_ID,
    PLAY_CARD_EPISODE_TIME_S,
    PLAY_CARD_MODEL_PATH,
    ROBOT_ID,
    ROBOT_PORT,
    WRIST_CAMERA_INDEX,
)


class RobotInterface(ABC):

    @abstractmethod
    def play_card(self) -> bool:
        """Pick up top card from deck and place it in the center arena."""

    @abstractmethod
    def collect_winnings(self) -> bool:
        """Grab both cards from the arena and move them to the score pile."""

    @abstractmethod
    def go_home(self) -> bool:
        """Return arm to neutral home position (called when robot loses)."""


# ── Mock ───────────────────────────────────────────────────────────────────────

class MockRobotInterface(RobotInterface):
    """Simulates robot actions with short sleeps — no hardware needed."""

    def play_card(self) -> bool:
        print("  [MOCK] Robot plays card → arena")
        time.sleep(1.0)
        return True

    def collect_winnings(self) -> bool:
        print("  [MOCK] Robot collects winnings → score pile")
        time.sleep(1.0)
        return True

    def go_home(self) -> bool:
        print("  [MOCK] Robot returns home")
        time.sleep(0.5)
        return True


# ── LeRobot ────────────────────────────────────────────────────────────────────

def _build_lerobot_record_cmd(
    policy_path: str,
    task_name: str,
    episode_time_s: int,
    _episode_index: int = 0,
) -> list[str]:
    """Build the lerobot-record command for one episode of a trained policy."""
    camera_cfg = (
        f"{{front: {{type: opencv, index_or_path: {WRIST_CAMERA_INDEX},"
        f" width: 640, height: 480, fps: 30}}}}"
    )
    # repo_id gets a suffix per skill to keep logs separate
    repo_id = f"{GAME_LOG_REPO_ID}/{task_name}"

    return [
        sys.executable, "-m", "lerobot.scripts.lerobot_record",
        # robot
        f"--robot.type=so100_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={camera_cfg}",
        # policy
        f"--policy.path={policy_path}",
        # dataset — local only, no HuggingFace push
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes=1",
        f"--dataset.single_task={task_name}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s=0",
        f"--dataset.push_to_hub=false",
    ]


def _run_skill(policy_path: str, task_name: str, episode_time_s: int) -> bool:
    """Run one skill episode. Blocks until complete. Returns True on success."""
    cmd = _build_lerobot_record_cmd(policy_path, task_name, episode_time_s)
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


class LeRobotInterface(RobotInterface):
    """
    Runs trained LeRobot policies via lerobot-record subprocess.
    Requires the 'lerobot' conda environment to be active.
    """

    def __init__(self):
        if PLAY_CARD_MODEL_PATH is None or COLLECT_WINNINGS_MODEL_PATH is None:
            raise ValueError(
                "PLAY_CARD_MODEL_PATH and COLLECT_WINNINGS_MODEL_PATH must be set in config.py "
                "before using LeRobotInterface."
            )
        self._episode_count = 0

    def play_card(self) -> bool:
        assert PLAY_CARD_MODEL_PATH is not None
        return _run_skill(PLAY_CARD_MODEL_PATH, "play_card", PLAY_CARD_EPISODE_TIME_S)

    def collect_winnings(self) -> bool:
        assert COLLECT_WINNINGS_MODEL_PATH is not None
        return _run_skill(COLLECT_WINNINGS_MODEL_PATH, "collect_winnings", COLLECT_WINNINGS_EPISODE_TIME_S)

    def go_home(self) -> bool:
        # "Going home" after a loss means staying still — the policy already
        # trained to end at the home position. Nothing to do here.
        # If you want an explicit home move, add a third trained policy.
        return True
