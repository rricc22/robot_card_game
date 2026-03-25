# ── Cameras ───────────────────────────────────────────────────────────────────
# Camera on the gripper of the follower arm (used during training, /dev/video4 from session log)
WRIST_CAMERA_INDEX = "/dev/video4"

# Optional overhead/top-down camera. Set to None if not connected.
OVERHEAD_CAMERA_INDEX = None   # e.g. "/dev/video2" once you add it

# Camera used to read cards in the arena.
# Prefer overhead; fall back to wrist if overhead not connected.
READ_CAMERA_INDEX = OVERHEAD_CAMERA_INDEX if OVERHEAD_CAMERA_INDEX is not None else WRIST_CAMERA_INDEX

# ── Timing ────────────────────────────────────────────────────────────────────
WAIT_AFTER_PLAY = 3.0    # seconds to wait after arm plays before reading cards
CARD_READ_ATTEMPTS = 3   # how many camera reads to attempt before giving up

# ── Robot hardware ────────────────────────────────────────────────────────────
ROBOT_PORT = "/dev/ttyACM0"    # follower arm serial port
ROBOT_ID   = "follower"

# ── LeRobot skill configuration ───────────────────────────────────────────────
# How long each skill episode runs (seconds). Tune to match your trained demos.
PLAY_CARD_EPISODE_TIME_S      = 10
COLLECT_WINNINGS_EPISODE_TIME_S = 10

# Model paths: HuggingFace repo IDs OR local paths like
#   "outputs/train/play_card/checkpoints/last/pretrained_model"
# Leave as None until models are trained — MockRobotInterface will be used automatically.
PLAY_CARD_MODEL_PATH      = None   # e.g. "rricc22/act_play_card_10k"
COLLECT_WINNINGS_MODEL_PATH = None  # e.g. "rricc22/act_collect_winnings_10k"

# Local directory to store game episode logs (no HuggingFace push needed)
GAME_LOG_REPO_ID = "local/card_game_log"
