import os
import sys


def load_config(path: str = "config.toml") -> dict:
    """Read *path* and return its contents as a nested dict."""
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: Configuration file '{path}' not found.\n"
            "       Create it or point to one with --config <path>."
        )

    try:
        import tomllib                      # Python >= 3.11
    except ModuleNotFoundError:
        try:
            import tomli as tomllib         # pip install tomli
        except ModuleNotFoundError:
            sys.exit(
                "ERROR: TOML support is unavailable.\n"
                "       Python >= 3.11 includes tomllib automatically.\n"
                "       For Python 3.9/3.10 run:  pip install tomli"
            )

    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        sys.exit(f"ERROR: Could not parse '{path}':\n  {exc}")
