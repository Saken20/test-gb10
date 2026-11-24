# Lets tests import bench/adapters from the repo root whatever directory pytest starts in.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
