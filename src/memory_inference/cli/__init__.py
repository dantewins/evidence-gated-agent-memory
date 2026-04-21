from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from memory_inference.cli.main import main as run_main

    run_main(argv)

__all__ = ["main"]
