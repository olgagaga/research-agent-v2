#!/usr/bin/env python3
"""Entry-point for the LLM-driven optimization agent.

Usage::

    python main.py [--max-iterations N]

Environment variables (see ``agent/config.py`` for defaults):

    MODEL_DIR          path to the mutable model repository
    MAIN_MODEL         OpenRouter model id (default: anthropic/claude-sonnet-4.6)
    TARGET_METRIC      ClearML metric to optimize (default: validation/mIoU)
    STATISTICAL_DELTA  relative-change threshold (default: 0.03)
"""

from __future__ import annotations

import argparse
import logging
import sys

from agent import run_loop
from agent import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-driven ML optimization agent",
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=None,
        help="Stop after N iterations (default: run forever)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the model directory (default: MODEL_DIR env or ./model_dir)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_dir = args.model_dir if args.model_dir else cfg.MODEL_DIR

    try:
        run_loop(model_dir=model_dir, max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        logging.getLogger(__name__).exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
