"""PyInstaller entry point for the standalone collector companion."""

from telemetry.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
