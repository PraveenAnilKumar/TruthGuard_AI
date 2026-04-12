import os
import sys

try:
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
except ImportError:
    print("sklearn NOT_INSTALLED")
except Exception as e:
    print(f"Error checking sklearn: {e}")
