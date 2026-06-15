"""Allow running as: python -m georsct.healthcheck <folder>"""

import sys

from .cli import main

sys.exit(main())
