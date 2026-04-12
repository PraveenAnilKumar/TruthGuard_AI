import os
import shutil
import sys
import gc
from pathlib import Path

def clear_python_cache():
    """Clear all __pycache__ directories recursively."""
    print("🧹 Clearing Python __pycache__ directories...")
    count = 0
    for p in Path('.').rglob('__pycache__'):
        try:
            shutil.rmtree(p)
            count += 1
        except Exception as e:
            print(f"  ⚠️ Could not remove {p}: {e}")
    print(f"✅ Removed {count} __pycache__ directories.")

def clear_streamlit_cache():
    """Clear Streamlit's local cache on disk."""
    print("🧹 Clearing Streamlit local cache...")
    st_cache = Path.home() / ".streamlit" / "cache"
    if st_cache.exists():
        try:
            shutil.rmtree(st_cache)
            print("✅ Streamlit disk cache cleared.")
        except Exception as e:
            print(f"  ⚠️ Could not remove {st_cache}: {e}")
    else:
        print("ℹ️ Streamlit disk cache not found (it might be empty).")

def main():
    print("=== TruthGuard AI Cache Cleaner ===")
    clear_python_cache()
    clear_streamlit_cache()
    
    print("\n💡 NOTE: To clear the active RAM cache while the app is running:")
    print("1. Use the new 🗑️ button in the TruthGuard AI sidebar/admin panel.")
    print("2. Or press 'C' inside the Streamlit browser window.")
    print("3. Or restart the application process.")
    
    print("\n✅ System cleanup complete.")

if __name__ == "__main__":
    main()
