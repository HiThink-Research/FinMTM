"""Backward-compatible entry point for objective evaluation."""

try:
    from .etest import main
except ImportError:
    from etest import main


if __name__ == "__main__":
    main()
