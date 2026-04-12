import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

CACHE_FOLDERS = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache"
]

CACHE_EXTENSIONS = [
    ".pyc",
    ".pyo"
]

TEMP_FOLDERS = [
    "temp"
]


def remove_cache_dirs():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for d in dirs:
            if d in CACHE_FOLDERS:
                path = Path(root) / d
                print(f"Removing cache folder: {path}")
                shutil.rmtree(path, ignore_errors=True)


def remove_cache_files():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for file in files:
            if any(file.endswith(ext) for ext in CACHE_EXTENSIONS):
                path = Path(root) / file
                print(f"Removing cache file: {path}")
                path.unlink(missing_ok=True)


def clean_temp():
    for folder in TEMP_FOLDERS:
        path = PROJECT_ROOT / folder
        if path.exists():
            print(f"Cleaning temp folder: {path}")
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)


def main():
    print("Cleaning project...")
    remove_cache_dirs()
    remove_cache_files()
    clean_temp()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()