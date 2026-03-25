"""Simple terminal UI using Rich."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.align import Align

console = Console()


def show_title():
    console.print()
    console.print(
        Panel(
            Align.center("[bold cyan]ROBOT CARD BATTLE[/bold cyan]\n[dim]SO-100 Arm  vs  Human[/dim]"),
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


def show_game_state(robot_score: int, human_score: int, robot_cards_left: int, human_cards_left: int):
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold dim")
    table.add_column("Player", style="bold", min_width=16)
    table.add_column("Score", justify="center", min_width=8)
    table.add_column("Cards left", justify="center", min_width=10)

    table.add_row("Robot", str(robot_score), str(robot_cards_left))
    table.add_row("Human", str(human_score), str(human_cards_left))

    console.print(table)


def show_round_header(round_num: int):
    console.rule(f"[bold white]Round {round_num}[/bold white]")


def show_cards_played(robot_card_str: str, human_card_str: str, result: int):
    """result: 1=robot wins, -1=human wins, 0=tie"""
    if result == 1:
        outcome = "[bold green]ROBOT WINS[/bold green]"
    elif result == -1:
        outcome = "[bold red]HUMAN WINS[/bold red]"
    else:
        outcome = "[bold yellow]TIE — play again![/bold yellow]"

    console.print(
        f"  Robot [cyan]{robot_card_str:>3}[/cyan]  vs  Human [magenta]{human_card_str:>3}[/magenta]"
        f"    {outcome}"
    )


def show_game_over(robot_score: int, human_score: int):
    console.print()
    if robot_score > human_score:
        msg = "[bold green]ROBOT WINS THE GAME![/bold green]"
        style = "green"
    elif human_score > robot_score:
        msg = "[bold red]HUMAN WINS THE GAME![/bold red]"
        style = "red"
    else:
        msg = "[bold yellow]IT'S A DRAW![/bold yellow]"
        style = "yellow"

    console.print(
        Panel(
            Align.center(f"{msg}\n\n[dim]Robot {robot_score}  —  Human {human_score}[/dim]"),
            border_style=style,
            padding=(1, 4),
        )
    )


def prompt(msg: str) -> None:
    console.print(f"\n[dim]{msg}[/dim]")
    input()


def status(msg: str) -> None:
    console.print(f"[dim]  {msg}[/dim]")


def warn(msg: str) -> None:
    console.print(f"[yellow]  ! {msg}[/yellow]")


def error(msg: str) -> None:
    console.print(f"[red]  ✗ {msg}[/red]")
