"""Allow running as: python -m failure_taxonomy <folder>"""

import sys

from .cli import main

sys.exit(main())
