"""Compatibility entrypoint; asset handling lives in the installed harness."""

from vla_eval.assets import *  # noqa: F403
from vla_eval.assets import main

if __name__ == "__main__":
    main()
