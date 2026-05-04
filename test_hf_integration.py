import unittest
from unittest.mock import patch

import numpy as np

from deepfake_detector_advanced import DeepfakeDetectorAdvanced


class DeepfakeHFIntegrationTests(unittest.TestCase):
    def test_hf_deepfake_requests_fall_back_when_remote_models_are_disabled(self):
        detector = DeepfakeDetectorAdvanced()
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        with patch.object(detector, "_ensure_hf_loaded", return_value=None), \
             patch.object(detector, "_get_loaded_model_subset", return_value=([], [])), \
             patch.object(detector, "detect_faces", return_value=[]):
            result = detector.detect_deepfake_ensemble(
                dummy_img,
                requested_models=["HF_Transformer_V1"],
            )

        self.assertIsInstance(result, dict)
        self.assertIn("is_deepfake", result)
        self.assertIn("ensemble_score", result)
        self.assertFalse(detector.enable_hf_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
