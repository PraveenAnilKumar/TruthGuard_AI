"""
FaceForensics++ Preprocessing Script for TruthGuard AI
Extracts face crops from FF++ videos and merges them into the
existing datasets/deepfake directory structure.

Usage:
    python prepare_ff_plus_plus.py
"""

import cv2
import os
import glob
from tqdm import tqdm
import logging
import random
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
FF_RAW_DIR      = "D:/TruthGuard_AI/datasets/ff_raw"
OUTPUT_DIR      = "D:/TruthGuard_AI/datasets/deepfake"
CROPS_PER_VIDEO = 10   # Face frames to extract per video
MAX_REAL_VIDEOS = 2000 # Cap real videos
MAX_FAKE_VIDEOS = 1500 # Cap per fake category

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── FF++ folder → label mapping ──────────────────────────────────────────────
FF_REAL_DIRS = [
    "original_sequences/actors/c40/videos",
    "original_sequences/youtube/c40/videos",
]

FF_FAKE_DIRS = [
    "manipulated_sequences/Deepfakes/c40/videos",
    "manipulated_sequences/Face2Face/c40/videos",
    "manipulated_sequences/FaceSwap/c40/videos",
    "manipulated_sequences/NeuralTextures/c40/videos",
    "manipulated_sequences/DeepFakeDetection/c40/videos",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def setup_dirs():
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

def extract_faces_from_video(video_path: str, output_folder: str,
                              label: str, video_id: str, split: str) -> int:
    """
    Extract up to CROPS_PER_VIDEO face crops from a single video.
    Returns count of successfully saved crops.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    step = max(total_frames // CROPS_PER_VIDEO, 1)
    saved = 0

    for i in range(CROPS_PER_VIDEO):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        margin  = int(w * 0.1)
        y1, y2  = max(0, y - margin), min(frame.shape[0], y + h + margin)
        x1, x2  = max(0, x - margin), min(frame.shape[1], x + w + margin)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Resize to a standard 224×224 to keep consistent with training expectations
        crop = cv2.resize(crop, (224, 224))

        save_name = f"ff_{label}_{video_id}_f{frame_idx}.jpg"
        save_path = os.path.join(OUTPUT_DIR, split, label, save_name)
        cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


def process_directory(video_dir: str, label: str, max_videos: int) -> int:
    full_dir = os.path.join(FF_RAW_DIR, video_dir)
    if not os.path.exists(full_dir):
        logger.warning(f"Directory not found, skipping: {full_dir}")
        return 0

    video_paths = glob.glob(os.path.join(full_dir, "*.mp4"))
    if not video_paths:
        logger.warning(f"No .mp4 files found in: {full_dir}")
        return 0

    # Shuffle so each training run sees variety even at lower caps
    random.shuffle(video_paths)
    video_paths = video_paths[:max_videos]

    logger.info(f"Processing {len(video_paths)} {label} videos from: {video_dir}")
    total_crops = 0

    for path in tqdm(video_paths, desc=f"  {label}/{Path(video_dir).parts[0]}"):
        video_id = Path(path).stem

        # Use official FF++ test split heuristic: videos ending in _000 go to test
        split = 'test' if video_id.endswith('_000') else 'train'

        crops = extract_faces_from_video(path, OUTPUT_DIR, label, video_id, split)
        total_crops += crops

    return total_crops


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("FaceForensics++ Preprocessing Pipeline")
    logger.info("Merging into: " + OUTPUT_DIR)
    logger.info("=" * 60)

    setup_dirs()

    grand_total = {"real": 0, "fake": 0}

    # ── Real videos ────────────────────────────────────────────────────────────
    logger.info("\n[1/2] Extracting REAL face crops ...")
    for real_dir in FF_REAL_DIRS:
        count = process_directory(real_dir, "real", MAX_REAL_VIDEOS)
        grand_total["real"] += count
        logger.info(f"  → {count} crops added from {real_dir}")

    # ── Fake videos ────────────────────────────────────────────────────────────
    logger.info("\n[2/2] Extracting FAKE face crops ...")
    for fake_dir in FF_FAKE_DIRS:
        count = process_directory(fake_dir, "fake", MAX_FAKE_VIDEOS)
        grand_total["fake"] += count
        logger.info(f"  → {count} crops added from {fake_dir}")

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ FaceForensics++ preprocessing complete!")
    logger.info(f"   Real crops added : {grand_total['real']:,}")
    logger.info(f"   Fake crops added : {grand_total['fake']:,}")
    total = grand_total['real'] + grand_total['fake']
    logger.info(f"   Total new images : {total:,}")
    logger.info("\nYour dataset now contains both Celeb-DF and FF++ data.")
    logger.info("Re-run training with:")
    logger.info("  python train_deepfake.py --model-type efficientnet "
                "--epochs 20 --batch-size 8 --steps-per-epoch 700")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
