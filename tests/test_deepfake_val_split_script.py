import tempfile
import unittest
from pathlib import Path

from scripts.create_deepfake_val_split import build_split_plan, derive_group_key


class DeepfakeValSplitScriptTests(unittest.TestCase):
    def test_derive_group_key_removes_frame_suffixes(self):
        self.assertEqual(
            derive_group_key("ff_fake_01_02__hugging_happy__YVGY8LOK_f171.jpg"),
            "ff_fake_01_02__hugging_happy__YVGY8LOK",
        )
        self.assertEqual(derive_group_key("clip_frame012.jpg"), "clip")
        self.assertEqual(derive_group_key("00000.jpg"), "00000")

    def test_build_split_plan_keeps_full_group_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            for rel_path in [
                "train/real/ff_real_scene_f0.jpg",
                "train/real/ff_real_scene_f30.jpg",
                "train/real/solo.jpg",
                "train/fake/ff_fake_scene_f0.jpg",
                "train/fake/ff_fake_scene_f30.jpg",
                "train/fake/fake_single.jpg",
            ]:
                path = dataset_dir / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            plan = build_split_plan(dataset_dir, val_ratio=0.34, seed=42)

            moved_real_names = {
                Path(move["source"]).name
                for move in plan["real"]["planned_moves"]
            }
            moved_fake_names = {
                Path(move["source"]).name
                for move in plan["fake"]["planned_moves"]
            }

            self.assertIn(
                moved_real_names,
                [
                    {"ff_real_scene_f0.jpg", "ff_real_scene_f30.jpg"},
                    {"solo.jpg"},
                ],
            )
            self.assertIn(
                moved_fake_names,
                [
                    {"ff_fake_scene_f0.jpg", "ff_fake_scene_f30.jpg"},
                    {"fake_single.jpg"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
