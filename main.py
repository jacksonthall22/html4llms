#!/usr/bin/env python3
"""
Compatibility entry point for running the tool from a checkout.

Prefer the installed CLI `html4llms`, but `python main.py` / `uv run main.py`
continue to work by delegating to `html4llms.cli:main`.
"""

from html4llms.cli import main


if __name__ == "__main__":
    main()
