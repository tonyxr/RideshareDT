import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch  # noqa: F401 -- missing real PyTorch is a test failure, never a stub run
