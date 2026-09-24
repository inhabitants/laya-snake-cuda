# laya-snake-cuda

**English** · [Português](README.pt-BR.md) · [Site](https://www.sapiensinteticos.com/laya-snake-cuda)

The Snake demo from [laya-mlx](https://github.com/mizorewww/laya-mlx), running on an NVIDIA GPU (or just the CPU) with PyTorch, on Windows or Linux. Plus one thing the original doesn't have: you can disturb the snake with the keyboard and watch it get back on track.

![Laya playing Snake on an RTX 3070 while being disturbed: the panel shows YOU pushing it, DODGED, and then recovered in 11 moves](docs/demo.gif)

*A real recorded run on an RTX 3070, replayed at 1× with `laya-snake export`. The pushes here come from `--chaos`; live, they come from your keys.*

## Credit first

This is a port, and most of it is not ours.

- **The demo** (game rules, safety planner, prompts, terminal UI, game loop) comes from [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx), Apache-2.0, which runs on Apple Silicon with MLX.
- **The model and its runtime** are [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache-2.0. The weights come from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) on Hugging Face and are not included here.

What this repo adds:

1. Runs on PyTorch (CUDA or CPU) through Convai's `laya` package, instead of MLX. Windows keyboard input included.
2. **Disturb mode**: WASD or the arrow keys push the snake off its path; when you let go, it steers back to the food.

Every derived file says at its top what was changed. See [NOTICE](NOTICE).

## What Laya does here, and what it doesn't

Laya does not generate text. Each move is one forward pass that answers three typed questions at once: which direction (a choice over four options), is there a safe route (yes/no), is the food reachable (yes/no). It returns probabilities, with zero output tokens.

The route itself is not Laya's work. A deterministic planner describes each of the four options in plain words ("Blocked", "Unsafe. Traps the snake", "Safe. Best route to food") and Laya picks one. An optional shield only lets safe moves through. That is the upstream design, and this port keeps it: the demo shows how fast and well-formed a typed decision is, not a model that plans on its own. `--unassisted` turns the shield off so you can see Laya's raw pick.

## Disturb mode

| Key | What it does |
|---|---|
| **W A S D** or **arrows** | Push the snake that way while you keep pressing. A tap lasts about 3 moves. |
| **+ / -** | Speed |
| **Space** | Pause |
| **R** | New round |
| **Q** | Quit |

The goal never changes: the snake is still after the food, you are just in the way. A push that would kill it is not obeyed (**DODGED**), and Laya takes that move instead.

When you let go, the panel reads **recovering...** and counts until the next food (**recovered in N moves**), or marks a **fall** if it dies first. The counters at the bottom add up disturbances, recoveries and falls.

Why a second planner exists: upstream, the snake never dies because it rides a fixed route that visits every cell of the board (a Hamiltonian cycle). A push breaks that route. While the snake is off it, the panel shows **FREE ROUTE** and a breadth-first planner computes the safe moves on the fly (a move is safe when the head can still reach the tail). Once the body is back in cycle order, it shows **FIXED ROUTE** and the upstream planner takes over again.

## Numbers we measured

| Hardware | Time per decision | Moves per second (`--max-speed`) |
|---|---|---|
| RTX 3070 8 GB (2020), CUDA | ~50 ms | ~18 to 20 |
| CPU only, Intel i7-11700 | ~270 to 330 ms | ~3 |

Multilingual checkpoint, 1.5 GB of VRAM at peak. The 3070 also drives the monitor; with another GPU job running at the same time we saw up to ~170 ms. For reference, laya-mlx reports 75.4 moves per second on an M3 Max with its optimized MLX path.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/inhabitants/laya-snake-cuda
cd laya-snake-cuda
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux), then:

```bash
# NVIDIA GPU: install a CUDA build of PyTorch first (RTX 50 series needs cu128 or newer)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
laya-snake download
laya-snake
```

`laya-snake download` fetches the multilingual checkpoint (~0.65 GB) into `./models/laya`, pinned to the revision this port was tested on. It writes plain files, because the Hugging Face cache uses symlinks, which Windows denies without admin rights.

**No GPU:** skip the torch line (pip installs the CPU build along with `laya`) and run `laya-snake --device cpu --fps 3`.

**Windows:** the board needs a terminal of at least 104 × 35. `scripts\play-windows.cmd` opens Windows Terminal at the right size.

## Options

| Flag | |
|---|---|
| `--device cpu` / `cuda:1` | Pick the device (default: CUDA when available) |
| `--lang pt` | Disturb-mode text in Portuguese (default: English) |
| `--unassisted` | Shield off: execute Laya's top pick as is |
| `--fps N` / `--max-speed` | Pace, or one move per finished inference |
| `--chaos N` | Automatic disturbance every N moves, for headless tests |
| `--headless --steps N` | Run without a display and print a JSON summary |
| `--record run.jsonl` | Save every decision and board state |
| `laya-snake export run.jsonl --output run.mp4 --gif run.gif` | Replay a recording at 1× as video and GIF (needs `pip install -e .[export]` and ffmpeg) |
| `--subfolder` | `multilingual` (default), `typed-decisions`, or `''` for the English root checkpoint |

## Status

Published as is, not maintained. Tested on Windows 11, Python 3.11, PyTorch 2.11 (CUDA 13), `laya` 0.3.5, RTX 3070. The Linux keyboard path follows the upstream termios code and was not tested here.

## License

Apache-2.0, like both upstream projects. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

Ported at [Sapiens Sintéticos](https://www.sapiensinteticos.com).
