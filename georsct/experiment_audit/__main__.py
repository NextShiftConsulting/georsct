"""Allow running as: python -m georsct.experiment_audit"""
import sys

from .cli import main

sys.exit(main())
