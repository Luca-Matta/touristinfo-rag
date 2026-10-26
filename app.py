"""Entry point: ``python app.py [question]``. See rag/cli.py for the options."""

import sys

from rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
