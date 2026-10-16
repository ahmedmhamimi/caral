"""
Allows `python -m caral` to start the CLI.

- (module) -> None: calls caral.cli.main() and exits with its return code.
"""

import sys

from .cli import main

sys.exit(main())
