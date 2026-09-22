"""Real Laya predictions, with an explicit optional deterministic safety shield.

Derived from laya_mlx/snake/policy.py (mizorewww/laya-mlx, Apache-2.0). Changed: the MLX
Agent became `laya.load(...)` from Convai's PyTorch runtime (CUDA or CPU, `device=`); the
checkpoint is a local folder filled by `laya-snake download` (the Hugging Face cache needs
symlinks, which Windows denies without admin); the metadata reads torch instead of mlx; and
the shield no longer halts when no move is safe (only possible after the player disturbed
the snake), it falls back to the roomiest legal move. The prompt and the questions are the
upstream ones.
"""

import hashlib
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .game import DIRECTIONS

HF_REPO = "convaiinnovations/laya"
HF_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"  # the revision this port was tested on
DEFAULT_DIR = Path("models") / "laya"
DEFAULT_SUBFOLDER = "multilingual"
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def download_checkpoint(directory=DEFAULT_DIR, subfolder=DEFAULT_SUBFOLDER):
    """Fetch one Laya checkpoint into a plain folder (no symlinks, works on Windows)."""
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    return snapshot_download(
        HF_REPO,
        revision=HF_REVISION,
        local_dir=str(directory),
        allow_patterns=[prefix + name for name in CHECKPOINT_FILES],
    )


def local_checkpoint(value=None, subfolder=DEFAULT_SUBFOLDER):
    """Resolve the local checkpoint folder; the game itself never touches the network."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    path = Path(value).expanduser() if value else DEFAULT_DIR
    if not (path / (subfolder or "") / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No Laya checkpoint in {path / (subfolder or '')}. Download it first (~0.65 GB):\n"
            f"  laya-snake download" + (f" --dir {value}" if value else "")
        )
    return path


def hardware_name(device):
    import torch

    if device.type == "cuda":
        return torch.cuda.get_device_name(device).removeprefix("NVIDIA ").removeprefix("GeForce ")
    return "CPU"


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def checkpoint_metadata(path, subfolder, agent):
    return {
        "name": f"{HF_REPO}/{subfolder}" if subfolder else HF_REPO,
        "revision": HF_REVISION,
        "hardware": hardware_name(agent.device),
        "engine": agent.device.type.upper(),
        "precision": (
            "FP32" if agent.device.type == "cpu"
            else "BF16" if "bfloat16" in str(agent.dtype) else "FP16"
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: _version(name) for name in ("laya", "torch", "transformers", "rich")},
        "network": "offline",
        "policy": "Laya probabilities over planner features; optional cycle safety shield",
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(self, model=None, *, subfolder=DEFAULT_SUBFOLDER, guarded=True, prompt="compact", device=None):
        import laya

        self.path = local_checkpoint(model, subfolder)
        self.agent = laya.load(str(self.path), device=device, subfolder=subfolder or None)
        self.guarded = guarded
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.prompt = prompt
        self.metadata = checkpoint_metadata(self.path, subfolder, self.agent)
        self.metadata["prompt"] = prompt

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe:
            # Only reachable after the player disturbed the snake: no move keeps the tail in
            # reach, so the shield falls back to the legal move with the most room.
            safe = [m for m in moves if m.legal]
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move toward food."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the snake."
                    if not m.safe
                    else "Safe. Eat food now. Best."
                    if m.eats
                    else "Safe. Best route to food."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["food"]["instructions"] = "Is food reachable through empty cells?"
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and allowed and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
