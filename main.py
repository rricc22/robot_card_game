"""
Robot Card Battle — main game loop.

Usage:
  python main.py              # auto: uses LeRobot if models are configured, else mock
  python main.py --mock       # force mock robot, real vision
  python main.py --mock --no-vision  # fully simulated (no camera needed)
"""

import sys
import time

from config import (
    READ_CAMERA_INDEX, WAIT_AFTER_PLAY, CARD_READ_ATTEMPTS,
    PLAY_CARD_MODEL_PATH, COLLECT_WINNINGS_MODEL_PATH,
)
from game_logic import Deck, Card, GameState, compare_cards
import display as ui

# ── flags ─────────────────────────────────────────────────────────────────────
_models_ready = PLAY_CARD_MODEL_PATH is not None and COLLECT_WINNINGS_MODEL_PATH is not None
USE_MOCK = "--mock" in sys.argv or not _models_ready
USE_VISION = "--no-vision" not in sys.argv

# ── robot ──────────────────────────────────────────────────────────────────────
if USE_MOCK:
    from robot_interface import MockRobotInterface
    robot = MockRobotInterface()
else:
    from robot_interface import LeRobotInterface
    robot = LeRobotInterface()

# ── vision ────────────────────────────────────────────────────────────────────
if USE_VISION:
    from vision import read_cards


def card_label(card: Card, ocr_rank: str | None) -> str:
    """Display label: use OCR rank if available, otherwise show deck card with a note."""
    if ocr_rank:
        return f"{ocr_rank}?"        # suit unknown from OCR
    return f"{card}*"               # * = from deck (OCR missed)


def play_round(state: GameState) -> None:
    """Play one round (may recurse on tie)."""
    state.round_number += 1
    ui.show_round_header(state.round_number)

    # Draw cards from each hand (physical deck tracking)
    robot_card = state.robot_hand.pop()
    human_card = state.human_hand.pop()

    # Robot physically plays its card
    ui.status("Robot playing card...")
    robot.play_card()

    # Human places their card
    ui.prompt("Place your card in the arena, then press ENTER")

    # Wait for everything to settle
    time.sleep(WAIT_AFTER_PLAY)

    # Read cards from camera
    ocr_robot, ocr_human = None, None
    if USE_VISION:
        ui.status("Reading cards from camera...")
        ocr_robot, ocr_human = read_cards(READ_CAMERA_INDEX, CARD_READ_ATTEMPTS)

        if not ocr_robot:
            ui.warn(f"Could not read robot card — using deck card ({robot_card})")
        if not ocr_human:
            ui.warn(f"Could not read human card — using deck card ({human_card})")

    # Resolve which card to use for comparison
    # If OCR succeeded, trust the camera; otherwise fall back to the deck draw
    def resolve(ocr_rank: str | None, deck_card: Card) -> Card:
        if ocr_rank:
            return Card(rank=ocr_rank)
        return deck_card

    played_robot = resolve(ocr_robot, robot_card)
    played_human = resolve(ocr_human, human_card)

    result = compare_cards(played_robot, played_human)
    ui.show_cards_played(card_label(robot_card, ocr_robot), card_label(human_card, ocr_human), result)

    if result == 1:
        state.robot_score += 1
        ui.status("Robot collecting winnings...")
        robot.collect_winnings()
    elif result == -1:
        state.human_score += 1
        ui.status("Robot going home...")
        robot.go_home()
    else:
        # Tie: play again immediately with next cards
        if state.game_over:
            ui.warn("Tie on the last card — it's a draw!")
            return
        play_round(state)   # recurse


def main():
    ui.show_title()

    if USE_MOCK:
        ui.console.print("[yellow]  Mock mode — no real robot[/yellow]")
    if not USE_VISION:
        ui.console.print("[yellow]  No-vision mode — using deck draws only[/yellow]")
    ui.console.print()

    # Deal deck
    deck = Deck()
    robot_hand, human_hand = deck.split()
    state = GameState(robot_hand=robot_hand, human_hand=human_hand)

    ui.console.print(f"  [green]26 cards each. Let's play![/green]\n")
    ui.show_game_state(state.robot_score, state.human_score, len(state.robot_hand), len(state.human_hand))

    try:
        while not state.game_over:
            play_round(state)
            ui.console.print()
            ui.show_game_state(
                state.robot_score, state.human_score,
                len(state.robot_hand), len(state.human_hand),
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        ui.console.print("\n[red]Game interrupted.[/red]")

    ui.show_game_over(state.robot_score, state.human_score)


if __name__ == "__main__":
    main()
