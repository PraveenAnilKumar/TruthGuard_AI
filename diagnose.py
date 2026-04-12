"""
TruthGuard AI Startup Diagnostic - Windows Safe Version
Run from your project folder:
    python diagnose.py

Results are also saved to: diagnose_results.txt
"""
import time
import sys
import os

# Force UTF-8 so Windows CMD does not swallow output
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

RESULTS = []

def log(msg):
    print(msg, flush=True)
    RESULTS.append(msg)

def test(label, fn, slow_threshold=2.0, note=""):
    print(f"  Testing {label}...", end="", flush=True)
    t = time.time()
    try:
        fn()
        elapsed = time.time() - t
        if elapsed > slow_threshold:
            suffix = f"  <-- SLOW ({elapsed:.2f}s){' [' + note + ']' if note else ''}"
            status = "SLOW" + suffix
        else:
            status = f"OK    ({elapsed:.2f}s)"
        line = f"  Testing {label}... {status}"
        print(f"\r{line:<80}")
        RESULTS.append(line)
        return elapsed
    except Exception as e:
        elapsed = time.time() - t
        line = f"  Testing {label}... FAILED ({elapsed:.2f}s): {type(e).__name__}: {e}"
        print(f"\r{line:<80}")
        RESULTS.append(line)
        return None


log("")
log("=" * 65)
log("  TruthGuard AI -- Startup Diagnostic")
log("  Python : " + sys.version.split()[0])
log("  Folder : " + os.getcwd())
log("=" * 65)

log("")
log("--- 1. Standard / Fast Libraries ---")
log("  Note: pandas >2s and sklearn >5s on first import is NORMAL.")
log("  They are only slow the very first run (no .pyc cache yet).")
log("  After first run they cache and become fast. Not a blocker.")
log("")
test("os / sys / json / hashlib",   lambda: [__import__(m) for m in ['os','sys','json','hashlib','re','logging']])
test("numpy",                       lambda: __import__('numpy'))
test("pandas",                      lambda: __import__('pandas'),   slow_threshold=5.0, note="normal on first run, not a blocker")
test("cv2 (OpenCV)",                lambda: __import__('cv2'))
test("PIL (Pillow)",                lambda: __import__('PIL'))
test("plotly",                      lambda: __import__('plotly'))
test("bcrypt",                      lambda: __import__('bcrypt'))
test("python-dotenv",               lambda: __import__('dotenv'))
test("sklearn",                     lambda: __import__('sklearn'),  slow_threshold=8.0, note="normal on first run, not a blocker")
test("joblib",                      lambda: __import__('joblib'))
test("nltk",                        lambda: __import__('nltk'),     slow_threshold=3.0)
test("requests",                    lambda: __import__('requests'))
test("tqdm",                        lambda: __import__('tqdm'))

log("")
log("--- 2. Optional / Heavy Libraries ---")
test("langdetect",                  lambda: __import__('langdetect'))

# googletrans: test import AND instantiation separately
print("  Testing googletrans import... ", end="", flush=True)
t = time.time()
try:
    from googletrans import Translator
    import_time = time.time() - t
    line = f"  Testing googletrans import... OK ({import_time:.2f}s)"
    print(f"\r{line:<80}")
    RESULTS.append(line)
    print("  Testing Translator()... ", end="", flush=True)
    t2 = time.time()
    tr = Translator()
    inst_time = time.time() - t2
    if inst_time > 2:
        line2 = f"  Testing Translator()... SLOW ({inst_time:.2f}s)  <-- BLOCKER: makes HTTP calls on init"
    else:
        line2 = f"  Testing Translator()... OK ({inst_time:.2f}s)"
    print(f"\r{line2:<80}")
    RESULTS.append(line2)
except Exception as e:
    elapsed = time.time() - t
    line = f"  Testing googletrans... FAILED ({elapsed:.2f}s): {e}"
    print(f"\r{line:<80}")
    RESULTS.append(line)

log("")
log("  --- transformers / torch ---")
log("  IMPORTANT: These are checked WITHOUT importing at module level.")
log("  If either CRASHES the process (no output after this line),")
log("  that is the bug. Our fixed files avoid this entirely.")
log("")

import importlib.util
tf_spec   = importlib.util.find_spec("transformers")
torch_spec = importlib.util.find_spec("torch")
log(f"  transformers installed: {'YES' if tf_spec else 'NO (pip install transformers)'}")
log(f"  torch installed       : {'YES' if torch_spec else 'NO (pip install torch)'}")

log("")
log("--- 3. TensorFlow (10-60s on first cold Windows run is normal) ---")
print("  Testing tensorflow... ", end="", flush=True)
t = time.time()
try:
    import tensorflow as tf
    elapsed = time.time() - t
    if elapsed > 30:
        line = f"  Testing tensorflow... VERY SLOW ({elapsed:.2f}s)  [normal once, but must NOT happen at every page load]"
    elif elapsed > 10:
        line = f"  Testing tensorflow... SLOW ({elapsed:.2f}s)  [acceptable cold start]"
    else:
        line = f"  Testing tensorflow... OK ({elapsed:.2f}s)"
    print(f"\r{line:<80}")
    RESULTS.append(line)
except Exception as e:
    elapsed = time.time() - t
    line = f"  Testing tensorflow... FAILED ({elapsed:.2f}s): {e}"
    print(f"\r{line:<80}")
    RESULTS.append(line)

log("")
log("--- 4. TruthGuard Singletons (ALL must be <1s with updated files) ---")
log("  These test whether the updated files are in place.")
log("  Any result >1s means that file was NOT replaced yet.")
log("")

module_checks = [
    ("translator_utils",   "from translator_utils import content_translator",           1.0),
    ("sentiment_analyzer", "from sentiment_analyzer import sentiment_analyzer",         1.0),
    ("fake_news_detector", "from fake_news_detector import fake_news_detector",         1.0),
    ("toxicity_detector",  "from toxicity_detector import toxicity_detector",           1.0),
    ("realtime_verifier",  "from realtime_verifier import realtime_verifier",           1.0),
    ("deepfake_detector",  "from deepfake_detector_advanced import deepfake_detector",  2.0),
]

for label, stmt, threshold in module_checks:
    print(f"  Testing {label}... ", end="", flush=True)
    t = time.time()
    try:
        exec(stmt, {})
        elapsed = time.time() - t
        if elapsed > threshold:
            line = f"  Testing {label}... SLOW ({elapsed:.2f}s)  <-- FILE NOT REPLACED"
        else:
            line = f"  Testing {label}... OK ({elapsed:.2f}s)"
        print(f"\r{line:<80}")
        RESULTS.append(line)
    except Exception as e:
        elapsed = time.time() - t
        line = f"  Testing {label}... FAILED ({elapsed:.2f}s): {type(e).__name__}: {e}"
        print(f"\r{line:<80}")
        RESULTS.append(line)

log("")
log("--- 5. .env File Check ---")
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    log("  .env file... FOUND")
    with open(env_path) as f:
        content = f.read()
    lines = [l for l in content.splitlines() if l.startswith('ADMIN_PASSWORD=')]
    if lines and lines[0].strip() != 'ADMIN_PASSWORD=':
        log("  ADMIN_PASSWORD... SET")
    else:
        log("  ADMIN_PASSWORD... WARNING: empty -- admin login disabled")
else:
    log("  .env file... WARNING: missing -- admin login disabled")
    log("               Create .env with: ADMIN_USER=admin  ADMIN_PASSWORD=yourpassword")

log("")
log("--- 6. Models Directory ---")
models_dir = os.path.join(os.getcwd(), 'models')
if os.path.exists(models_dir):
    h5_files = []
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith('.h5') or f.endswith('.weights.h5'):
                full = os.path.join(root, f)
                size_mb = os.path.getsize(full) / (1024 * 1024)
                h5_files.append((os.path.relpath(full, os.getcwd()), size_mb))
    if h5_files:
        log(f"  Found {len(h5_files)} .h5 model file(s):")
        for path, size in h5_files:
            log(f"    {path}  ({size:.1f} MB)")
    else:
        log("  No .h5 files found -- deepfake will use pretrained ImageNet fallback on first use")
else:
    log("  models/ directory not found -- will be created on first run")

log("")
log("=" * 65)
log("  SUMMARY")
log("=" * 65)

real_blockers = [r for r in RESULTS if
    ('BLOCKER' in r or 'FILE NOT REPLACED' in r or 'FAILED' in r)
    and 'normal' not in r.lower()
    and 'acceptable' not in r.lower()]
warnings = [r for r in RESULTS if 'WARNING' in r]

if real_blockers:
    log(f"  {len(real_blockers)} BLOCKER(S) found -- fix these before starting:")
    for b in real_blockers:
        log(f"    >> {b.strip()}")
else:
    log("  No blockers detected.")

if warnings:
    log(f"  {len(warnings)} warning(s):")
    for w in warnings:
        log(f"    >> {w.strip()}")

if not real_blockers and not warnings:
    log("  Everything looks good -- run: python -m streamlit run app.py")

log("")
log("  Full results saved to: diagnose_results.txt")
log("=" * 65)

try:
    with open("diagnose_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(RESULTS))
except Exception as e:
    print(f"  Could not write results file: {e}")