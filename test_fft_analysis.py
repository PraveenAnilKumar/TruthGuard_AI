import unittest

import numpy as np

from deepfake_detector_advanced import DeepfakeDetectorAdvanced


class DeepfakeFFTTests(unittest.TestCase):
    def test_fft_analysis_handles_natural_and_artifact_patterns(self):
        detector = DeepfakeDetectorAdvanced()

        natural_img = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            natural_img[:, i, :] = i

        artifact_img = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 4):
            for j in range(0, 224, 4):
                artifact_img[i:i + 2, j:j + 2, :] = 255

        _, natural_score = detector.perform_fft_analysis(natural_img)
        _, artifact_score = detector.perform_fft_analysis(artifact_img)

        self.assertGreaterEqual(natural_score, 0.0)
        self.assertLessEqual(natural_score, 1.0)
        self.assertGreaterEqual(artifact_score, 0.0)
        self.assertLessEqual(artifact_score, 1.0)
        self.assertGreaterEqual(artifact_score, natural_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
