import os
import sys

import numpy as np

# Add current directory to path
sys.path.insert(0, os.getcwd())


def test_fft_analysis():
    print("=" * 50)
    print("Testing FFT Analysis")
    print("=" * 50)

    print("\nInitializing detector for FFT forensic test...")
    try:
        from deepfake_detector_advanced import DeepfakeDetectorAdvanced
        detector = DeepfakeDetectorAdvanced()
        print("Detector initialized")
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return 1

    natural_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        natural_img[:, i, :] = i

    artifact_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 4):
        for j in range(0, 224, 4):
            artifact_img[i:i + 2, j:j + 2, :] = 255

    print("\nAnalyzing NATURAL image...")
    try:
        _, score_nat = detector.perform_fft_analysis(natural_img)
        print(f"Natural FFT Score: {score_nat:.4f}")
    except Exception as e:
        print(f"Error analyzing natural image: {e}")
        return 1

    print("\nAnalyzing ARTIFACT image...")
    try:
        _, score_art = detector.perform_fft_analysis(artifact_img)
        print(f"Artifact FFT Score: {score_art:.4f}")
    except Exception as e:
        print(f"Error analyzing artifact image: {e}")
        return 1

    print("\n" + "=" * 50)
    if score_art >= score_nat:
        print("FFT analysis completed successfully.")
        print("=" * 50)
        return 0

    print("FFT analysis did not distinguish the synthetic artifact pattern.")
    print("=" * 50)
    return 1


if __name__ == "__main__":
    raise SystemExit(test_fft_analysis())
