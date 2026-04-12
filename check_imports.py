import sys
import time
print("Checking imports...")
sys.stdout.flush()
t0 = time.time()
import numpy
print(f"NumPy imported in {time.time()-t0:.2f}s")
sys.stdout.flush()

t0 = time.time()
import torch
print(f"Torch imported in {time.time()-t0:.2f}s")
sys.stdout.flush()

t0 = time.time()
from transformers import pipeline
print(f"Transformers pipeline imported in {time.time()-t0:.2f}s")
sys.stdout.flush()

print("All imports done.")
