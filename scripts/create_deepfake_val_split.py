"""
Create a leakage-resistant validation split for the deepfake dataset.

The script groups obvious multi-frame crops by source video stem so FF++ style
files such as `ff_real_scene_f120.jpg` stay together. Files without an obvious
frame suffix are treated as single-sample groups.

By default the script performs a dry run. Use `--apply` to move files from
`train/` into `val/`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_SUFFIX_PATTERNS = [
    re.compile(r"^(?P<base>.+?)(?:[_-]f)(?P<index>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+?)(?:[_-]frame)(?P<index>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+?)(?:[_-]img)(?P<index>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+?)(?:[_-]image)(?P<index>\d+)$", re.IGNORECASE),
]


def derive_group_key(filename: str) -> str:
    """Collapse obvious frame suffixes so crops from the same source stay together."""
    stem = Path(filename).stem
    for pattern in FRAME_SUFFIX_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group("base")
    return stem


def list_image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def group_files_by_source(files: Iterable[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        grouped[derive_group_key(path.name)].append(path)
    return dict(grouped)


def choose_groups_for_target(grouped_files: Dict[str, List[Path]], target_count: int, seed: int) -> List[str]:
    """Pick whole source groups until the requested validation count is reached."""
    if target_count <= 0 or not grouped_files:
        return []

    items = list(grouped_files.items())
    rng = random.Random(seed)
    rng.shuffle(items)

    selected_groups: List[str] = []
    selected_count = 0
    for group_key, paths in items:
        if selected_count >= target_count:
            break
        selected_groups.append(group_key)
        selected_count += len(paths)
    return selected_groups


def plan_label_split(train_dir: Path, val_dir: Path, val_ratio: float, seed: int) -> Dict[str, object]:
    train_files = list_image_files(train_dir)
    existing_val_files = list_image_files(val_dir)
    total_files = len(train_files) + len(existing_val_files)
    target_val_count = int(round(total_files * val_ratio))
    additional_needed = max(0, target_val_count - len(existing_val_files))

    grouped_train_files = group_files_by_source(train_files)
    selected_groups = choose_groups_for_target(grouped_train_files, additional_needed, seed)

    planned_moves = []
    for group_key in selected_groups:
        for source_path in grouped_train_files[group_key]:
            planned_moves.append(
                {
                    "group_key": group_key,
                    "source": str(source_path),
                    "destination": str(val_dir / source_path.name),
                }
            )

    return {
        "train_count": len(train_files),
        "existing_val_count": len(existing_val_files),
        "target_val_count": target_val_count,
        "additional_needed": additional_needed,
        "selected_group_count": len(selected_groups),
        "selected_groups": selected_groups,
        "planned_moves": planned_moves,
        "planned_move_count": len(planned_moves),
    }


def build_split_plan(dataset_dir: Path, val_ratio: float, seed: int) -> Dict[str, Dict[str, object]]:
    plan = {}
    for label in ["real", "fake"]:
        plan[label] = plan_label_split(
            train_dir=dataset_dir / "train" / label,
            val_dir=dataset_dir / "val" / label,
            val_ratio=val_ratio,
            seed=seed,
        )
    return plan


def apply_plan(plan: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    moved_counts = {"real": 0, "fake": 0}
    for label, label_plan in plan.items():
        for move in label_plan["planned_moves"]:
            source = Path(move["source"])
            destination = Path(move["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists() or destination.exists():
                continue
            shutil.move(str(source), str(destination))
            moved_counts[label] += 1
    return moved_counts


def summarize_plan(plan: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    summary = {}
    for label, label_plan in plan.items():
        summary[label] = {
            "train_count": int(label_plan["train_count"]),
            "existing_val_count": int(label_plan["existing_val_count"]),
            "target_val_count": int(label_plan["target_val_count"]),
            "planned_move_count": int(label_plan["planned_move_count"]),
            "selected_group_count": int(label_plan["selected_group_count"]),
        }
    return summary


def write_manifest(dataset_dir: Path, plan: Dict[str, Dict[str, object]]) -> Path:
    manifest_path = dataset_dir / "val_split_manifest.json"
    payload = {
        "dataset_dir": str(dataset_dir),
        "summary": summarize_plan(plan),
        "plan": plan,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a grouped validation split for deepfake training data.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/deepfake"), help="Deepfake dataset root directory.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Target validation ratio across train+val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source-group selection.")
    parser.add_argument("--apply", action="store_true", help="Move files from train/ into val/.")
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON.")
    args = parser.parse_args()

    dataset_dir = args.data_dir.resolve()
    plan = build_split_plan(dataset_dir=dataset_dir, val_ratio=args.val_ratio, seed=args.seed)

    if args.json:
        print(json.dumps({"dataset_dir": str(dataset_dir), "summary": summarize_plan(plan)}, indent=2))
    else:
        print(f"Dataset: {dataset_dir}")
        print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
        print(f"Validation ratio target: {args.val_ratio:.2%}")
        for label, label_summary in summarize_plan(plan).items():
            print(
                f"[{label}] train={label_summary['train_count']}, "
                f"existing_val={label_summary['existing_val_count']}, "
                f"target_val={label_summary['target_val_count']}, "
                f"planned_moves={label_summary['planned_move_count']}, "
                f"groups={label_summary['selected_group_count']}"
            )

    if args.apply:
        manifest_path = write_manifest(dataset_dir, plan)
        moved_counts = apply_plan(plan)
        print(
            "Moved files: "
            f"real={moved_counts['real']}, fake={moved_counts['fake']}, total={sum(moved_counts.values())}"
        )
        print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
