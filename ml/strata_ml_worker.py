"""Entry point: python strata_ml_worker.py --home <ml home>  (Strata starts this for you)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strata_ml.worker import main  # noqa: E402

if __name__ == "__main__":
    main()
