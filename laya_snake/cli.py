"""Watch Laya play Snake in the terminal, and disturb it to see it recover.

Derived from laya_mlx/snake/cli.py (mizorewww/laya-mlx, Apache-2.0). The game loop, pacing,
recording format and summary are the upstream ones. Changed: the keyboard reads msvcrt on
Windows (termios on Linux/macOS, as upstream) and returns key tokens; `--device`,
`--subfolder` and `--lang` were added; `--optimize` (MLX compile) and the benchmark
subcommand were left out; `download` is new and `export` (replay.py) stays; a dead round
always restarts (also with --unassisted); and the disturb mode below.

Keys
  W A S D or arrows   disturb: the snake is pushed that way while you keep pressing (a tap
                      lasts HOLD_SECONDS). The goal does not change, Laya is still after the
                      food. A push that would kill it is not obeyed: Laya takes that move
                      ("DODGED").
  + / -               speed      SPACE pause      R new round      Q quit

When you let go, Laya steers back and the panel reads "recovering..."; eating the next food
counts as recovered ("recovered in N moves"), dying first counts as a fall ("fell").
"""

import argparse
import json
import random
import sys
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live

from .game import DIRECTIONS, SnakeGame
from .i18n import STRINGS, text
from .policy import DEFAULT_DIR, DEFAULT_SUBFOLDER, LayaPolicy, download_checkpoint
from .ui import BG, compose, layout_size

ARROW_NAMES = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
PUSH_KEYS = {"w": "UP", "s": "DOWN", "a": "LEFT", "d": "RIGHT",
             "UP_ARROW": "UP", "DOWN_ARROW": "DOWN", "LEFT_ARROW": "LEFT", "RIGHT_ARROW": "RIGHT"}
FLASH_SECONDS = 1.2
HOLD_SECONDS = 0.3  # a tap is a nudge of ~3 moves at 12 fps; holding the key keeps nudging


class Keyboard:
    """Non-blocking key reads, returned as tokens (lowercase chars or *_ARROW names)."""

    WIN_ARROWS = {"H": "UP_ARROW", "P": "DOWN_ARROW", "K": "LEFT_ARROW", "M": "RIGHT_ARROW"}
    ANSI_ARROWS = {"A": "UP_ARROW", "B": "DOWN_ARROW", "D": "LEFT_ARROW", "C": "RIGHT_ARROW"}

    def __enter__(self):
        self.msvcrt = self.saved = None
        if not sys.stdin.isatty():
            return self
        try:
            import msvcrt

            self.msvcrt = msvcrt
        except ImportError:
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self):
        out = []
        if self.msvcrt:
            while self.msvcrt.kbhit():
                ch = self.msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    name = self.WIN_ARROWS.get(self.msvcrt.getwch())
                    if name:
                        out.append(name)
                else:
                    out.append(ch.lower())
        elif self.saved is not None:
            import os
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                raw = os.read(self.fd, 128).decode(errors="ignore")
                i = 0
                while i < len(raw):
                    if raw.startswith("\x1b[", i) and i + 2 < len(raw) and raw[i + 2] in self.ANSI_ARROWS:
                        out.append(self.ANSI_ARROWS[raw[i + 2]])
                        i += 3
                    else:
                        out.append(raw[i].lower())
                        i += 1
        return out

    def __exit__(self, *_):
        if self.saved is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def positive(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return result


def download(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-snake download",
        description="Download a Laya checkpoint (Convai Innovations, Apache-2.0) into a plain folder.",
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help=f"Target folder (default: {DEFAULT_DIR})")
    parser.add_argument(
        "--subfolder",
        default=DEFAULT_SUBFOLDER,
        help="multilingual (default, ~0.65 GB), typed-decisions, or '' for the English root (~0.84 GB each)",
    )
    args = parser.parse_args(argv)
    print(f"Downloading {args.subfolder or 'root'} checkpoint into {args.dir} ...", file=sys.stderr)
    download_checkpoint(args.dir, args.subfolder)
    print("Done. Run: laya-snake" + (f" --model {args.dir}" if args.dir != DEFAULT_DIR else ""))
    return 0


def play(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-snake",
        description=__doc__,
        epilog="First time: laya-snake download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help=f"Local checkpoint folder (default: {DEFAULT_DIR})")
    parser.add_argument(
        "--subfolder",
        default=DEFAULT_SUBFOLDER,
        help="Checkpoint inside the model folder: multilingual (default), typed-decisions, or '' for the English root",
    )
    parser.add_argument("--device", help="cuda, cuda:1, cpu... (default: CUDA when available, else CPU)")
    parser.add_argument("--lang", choices=sorted(STRINGS), default="en", help="Disturb-mode text (default: en)")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument(
        "--fps", type=positive, default=12, help="Game decisions per second (default: 12)"
    )
    parser.add_argument(
        "--max-speed", action="store_true", help="One move per completed inference, without pacing"
    )
    parser.add_argument(
        "--duration", type=positive, help="Stop after this many seconds, excluding warmup"
    )
    parser.add_argument("--steps", type=int, help="Stop after this many actual game steps")
    parser.add_argument(
        "--unassisted",
        action="store_true",
        help="Execute raw Laya top-1; disable the safety shield",
    )
    parser.add_argument(
        "--chaos",
        type=int,
        metavar="N",
        help="Automatic disturbance every N moves: a random direction held for HOLD_SECONDS",
    )
    parser.add_argument(
        "--record", type=Path, help="Write timestamped real decisions and board states to JSONL"
    )
    parser.add_argument("--headless", action="store_true", help="Run without a terminal display")
    parser.add_argument(
        "--no-alt-screen", action="store_true", help="Keep the final frame in terminal scrollback"
    )
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.chaos is not None and args.chaos < 1:
        parser.error("--chaos must be positive")
    try:
        game = SnakeGame(args.width, args.height, args.seed, args.initial_length)
    except ValueError as error:
        parser.error(str(error))
    console = Console(style=f"on {BG}", highlight=False)
    if not args.headless and not console.is_terminal:
        parser.error("Interactive display needs a TTY. Use --headless for a non-interactive run.")
    t = text(args.lang)
    print("Loading local weights; no network requests...", file=sys.stderr)
    try:
        policy = LayaPolicy(
            args.model,
            subfolder=args.subfolder,
            guarded=not args.unassisted,
            prompt=args.prompt,
            device=args.device,
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    warm = SnakeGame(args.width, args.height, args.seed + 10000, args.initial_length)
    for _ in range(6):
        decision = policy.decide(warm)
        warm.step(decision.executed)
        if not warm.alive:
            break
    record = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record = args.record.open("x")
        record.write(
            json.dumps(
                {
                    "type": "metadata",
                    "format": "laya-snake-v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": policy.metadata,
                    "settings": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                    "note": "Real synchronized inference; board is shown before the announced action. Risk is 1 - P(safe route).",
                }
            )
            + "\n"
        )
    started = time.perf_counter()
    stats = {
        "hardware": policy.metadata["hardware"],
        "engine": policy.metadata["engine"],
        "precision": policy.metadata["precision"],
        "lang": args.lang,
        "guarded": policy.guarded,
        "interventions": 0,
        "best": 0,
        "round": 1,
        "paused": False,
        "elapsed": 0,
        "steps_per_second": 0,
        "takeovers": 0,
        "recoveries": 0,
        "falls": 0,
        "driving": False,
        "status": None,
        "flash_text": "",
    }
    calls = total_steps = deaths = 0
    timestamps = deque(maxlen=60)
    inference = []
    displayed_board, displayed_decision = game.snapshot(), {}
    chaos_rng = random.Random(args.seed)
    steer, steer_until, steered = None, 0.0, 0  # direction being pushed, until when, moves obeyed
    recovering_since = None
    flash_until = 0.0

    def flash(message):
        nonlocal flash_until
        stats["flash_text"] = message
        flash_until = time.perf_counter() + FLASH_SECONDS

    def release(count=True):
        """The push is over: Laya steers back and the recovery clock starts."""
        nonlocal steer, steered, recovering_since
        if count and steered:
            stats["takeovers"] += 1
            recovering_since = total_steps
            stats["status"] = {"kind": "recovering"}
        steer, steered = None, 0
        stats["driving"] = False

    def start_push(direction, now):
        nonlocal steer, steer_until, recovering_since
        steer, steer_until = direction, now + HOLD_SECONDS
        stats["driving"], stats["status"] = True, None
        recovering_since = None

    live = (
        Live(
            console=console,
            screen=not args.no_alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        if not args.headless
        else None
    )
    try:
        with Keyboard() as keys, live if live else nullcontext():
            while True:
                now = time.perf_counter()
                if (args.duration and now - started >= args.duration) or (
                    args.steps and total_steps >= args.steps
                ):
                    break
                if now >= flash_until:
                    stats["flash_text"] = ""
                pressed = keys.read()
                if "q" in pressed or "\x03" in pressed:
                    break
                if " " in pressed:
                    stats["paused"] = not stats["paused"]
                if "+" in pressed or "=" in pressed:
                    args.fps = min(240, args.fps + 2)
                if "-" in pressed:
                    args.fps = max(1, args.fps - 2)
                if "r" in pressed:
                    stats["round"] += 1
                    game = SnakeGame(
                        args.width, args.height, args.seed + stats["round"] - 1, args.initial_length
                    )
                    displayed_board, displayed_decision = game.snapshot(), {}
                    release(count=False)
                    recovering_since, stats["status"] = None, None
                if not stats["paused"]:
                    for token in pressed:
                        if token in PUSH_KEYS:
                            start_push(PUSH_KEYS[token], now)
                    if args.chaos and total_steps and total_steps % args.chaos == 0 and not steer:
                        start_push(chaos_rng.choice(DIRECTIONS), now)
                    if steer and now >= steer_until:
                        release()
                if stats["paused"]:
                    stats["elapsed"] = now - started
                    if live:
                        live.update(
                            compose(displayed_board, displayed_decision, stats).rich_text(),
                            refresh=True,
                        )
                    time.sleep(0.03)
                    continue
                minimum_width, minimum_height = layout_size(game.width, game.height)
                if live and (console.width < minimum_width or console.height < minimum_height):
                    live.update(
                        f"Resize terminal to at least {minimum_width} columns × {minimum_height} rows.\n"
                        "The game is waiting. Q quits.",
                        refresh=True,
                    )
                    time.sleep(0.1)
                    continue
                decision = policy.decide(game)
                calls += 1
                inference.append(decision.inference_ms)
                pushed = False
                if steer:
                    if game.legal_reason(steer) == "legal":
                        decision.executed, decision.intervened = steer, False
                        pushed = True
                        steered += 1
                        flash(t["you"].format(arrow=ARROW_NAMES[steer]))
                    else:
                        flash(t["dodged"])
                stats["interventions"] += decision.intervened
                shown = time.perf_counter()
                timestamps.append(shown)
                stats["elapsed"] = shown - started
                stats["steps_per_second"] = (
                    (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                    if len(timestamps) > 1
                    else 0
                )
                stats["best"] = max(stats["best"], game.score)
                board = game.snapshot()
                shown_decision = {**decision.to_dict(), "pushed": pushed}
                displayed_board, displayed_decision = board, shown_decision
                if live:
                    canvas = compose(board, shown_decision, stats)
                    live.update(canvas.rich_text(), refresh=True)
                if record:
                    record.write(
                        json.dumps(
                            {
                                "type": "frame",
                                "at": shown - started,
                                "game": board,
                                "decision": shown_decision,
                                "stats": dict(stats),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                if not args.max_speed:
                    remaining = 1 / args.fps - (time.perf_counter() - now)
                    if remaining > 0:
                        time.sleep(remaining)
                ate = game.step(decision.executed)
                total_steps += 1
                stats["best"] = max(stats["best"], game.score)
                if ate and recovering_since is not None:
                    stats["recoveries"] += 1
                    stats["status"] = {"kind": "recovered", "moves": total_steps - recovering_since}
                    recovering_since = None
                if not game.alive or game.won:
                    deaths += not game.alive
                    if not game.alive and (recovering_since is not None or steered):
                        stats["falls"] += 1
                        stats["status"] = {"kind": "fell", "reason": game.death_reason}
                        recovering_since = None
                    release(count=False)
                    if record:
                        record.write(
                            json.dumps(
                                {
                                    "type": "round_end",
                                    "at": time.perf_counter() - started,
                                    "game": game.snapshot(),
                                }
                            )
                            + "\n"
                        )
                    if live:
                        live.update(compose(game.snapshot(), {}, stats).rich_text(), refresh=True)
                        time.sleep(1)
                    stats["round"] += 1
                    game = SnakeGame(
                        args.width, args.height, args.seed + stats["round"] - 1, args.initial_length
                    )
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - started
        summary = {
            "steps": total_steps,
            "inference_calls": calls,
            "seconds": elapsed,
            "steps_per_second": total_steps / elapsed if elapsed else 0,
            "score": game.score,
            "length": len(game.body),
            "best_score": stats["best"],
            "interventions": stats["interventions"],
            "deaths": deaths,
            "disturbances": stats["takeovers"],
            "recoveries": stats["recoveries"],
            "falls": stats["falls"],
            "guarded": policy.guarded,
            "device": policy.metadata["hardware"],
            "network": "offline",
            "mean_inference_ms": sum(inference) / len(inference) if inference else None,
        }
        if record:
            record.write(
                json.dumps({"type": "end", "summary": summary, "game": game.snapshot()}) + "\n"
            )
            record.close()
        print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "download":
        return download(argv[1:])
    if argv and argv[0] == "export":
        from .replay import main as export

        return export(argv[1:])
    return play(argv)
