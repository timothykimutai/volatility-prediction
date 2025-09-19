import sys
import os

# Ensure project root is on sys.path so 'src' imports work during tests
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
