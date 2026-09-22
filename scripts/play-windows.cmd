@echo off
rem Opens laya-snake in a Windows Terminal window big enough for the board (104x35 minimum).
rem Run it from the repo folder, with the virtual environment in .venv. Arguments pass through:
rem   scripts\play-windows.cmd --lang pt
rem   scripts\play-windows.cmd --device cpu --fps 3
start "" wt.exe --size 112,38 --title "laya-snake" -d "%~dp0.." cmd /c "set PYTHONUTF8=1&& .venv\Scripts\python.exe -m laya_snake %* & pause"
