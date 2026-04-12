import unittest
from unittest.mock import patch

import numpy as np

import safe_deepfake_runtime
from deepfake_detector_advanced import DeepfakeDetectorAdvanced


class _ScaledCascade:
    def __init__(self, base_boxes, original_width):
        self.base_boxes = base_boxes
        self.original_width = original_width

    def detectMultiScale(self, image, **kwargs):
        scale = image.shape[1] / float(self.original_width)
        return np.array(
            [
                [
                    int(round(x * scale)),
                    int(round(y * scale)),
                    int(round(w * scale)),
                    int(round(h * scale)),
                ]
                for x, y, w, h in self.base_boxes
            ],
            dtype=np.int32,
        )


class _EmptyCascade:
    def detectMultiScale(self, image, **kwargs):
        return np.empty((0, 4), dtype=np.int32)


class _FakeVideoCapture:
    def __init__(self, frames, fps=30.0):
        self.frames = list(frames)
        self.fps = fps
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        return None

    def get(self, prop_id):
        if prop_id == 5:
            return self.fps
        return 0.0


class _FakeGradCamModel:
    layers = [object()]


class DeepfakeDetectorUpgradeTests(unittest.TestCase):
    def _build_detector(self):
        with patch.object(
            DeepfakeDetectorAdvanced,
            "_load_existing_models",
            lambda self, requested_names=None: None,
        ), patch.object(
            DeepfakeDetectorAdvanced,
            "_discover_model_files",
            return_value=[],
        ):
            return DeepfakeDetectorAdvanced()

    def test_init_scans_models_without_loading_them(self):
        with patch.object(
            DeepfakeDetectorAdvanced,
            "_discover_model_files",
            return_value=["models/deepfake_best.h5", "models/deepfake_extra.weights.h5"],
        ):
            detector = DeepfakeDetectorAdvanced()

        self.assertEqual(detector.model_names, [])
        self.assertEqual(
            detector.available_model_names,
            ["deepfake_best", "deepfake_extra"],
        )
        self.assertEqual(detector.get_preferred_model_name(), "deepfake_best")

    def test_preferred_model_uses_highest_evaluated_accuracy_when_available(self):
        with patch.object(
            DeepfakeDetectorAdvanced,
            "_discover_model_files",
            return_value=[
                "models/deepfake_best_20260314_183448.h5",
                "models/deepfake_best_20260315_202503.weights.h5",
            ],
        ), patch.object(
            DeepfakeDetectorAdvanced,
            "_load_evaluation_accuracy_map",
            return_value={
                "20260314_221515": 0.89255,
                "20260316_010648": 0.91475,
            },
        ):
            detector = DeepfakeDetectorAdvanced()

        self.assertEqual(detector.get_preferred_model_name(), "deepfake_best_20260315_202503")

    def test_ensure_models_ready_defaults_to_one_preferred_model(self):
        detector = self._build_detector()
        detector.available_model_specs = [
            {"name": "deepfake_best", "path": "models/deepfake_best.h5", "kind": "full_model"},
            {"name": "deepfake_artifact", "path": "models/deepfake_artifact.h5", "kind": "full_model"},
            {"name": "deepfake_cnn", "path": "models/deepfake_cnn.h5", "kind": "full_model"},
        ]
        detector.available_model_names = [spec["name"] for spec in detector.available_model_specs]
        detector._model_scan_complete = True

        calls = []

        def fake_load(requested_names=None):
            calls.append(list(requested_names or []))

        detector._load_existing_models = fake_load

        with patch.object(detector, "_load_pretrained_models") as pretrained_mock:
            detector._ensure_models_ready()

        self.assertEqual(calls, [["deepfake_best"]])
        pretrained_mock.assert_not_called()

    def test_ensure_models_ready_expands_only_to_target_ensemble(self):
        detector = self._build_detector()
        detector.max_ensemble_local_models = 2
        detector.available_model_specs = [
            {"name": "deepfake_best", "path": "models/deepfake_best.h5", "kind": "full_model"},
            {"name": "deepfake_artifact", "path": "models/deepfake_artifact.h5", "kind": "full_model"},
            {"name": "deepfake_cnn", "path": "models/deepfake_cnn.h5", "kind": "full_model"},
        ]
        detector.available_model_names = [spec["name"] for spec in detector.available_model_specs]
        detector._model_scan_complete = True
        detector.model_names = ["deepfake_best"]

        calls = []

        def fake_load(requested_names=None):
            calls.append(list(requested_names or []))

        detector._load_existing_models = fake_load

        with patch.object(detector, "_load_pretrained_models") as pretrained_mock:
            detector._ensure_models_ready(prefer_ensemble=True)

        self.assertEqual(calls, [["deepfake_artifact"]])
        pretrained_mock.assert_not_called()

    def test_detect_deepfake_ensemble_ignores_previously_loaded_non_ensemble_models(self):
        detector = self._build_detector()
        detector.max_ensemble_local_models = 2
        detector.available_model_specs = [
            {"name": "deepfake_best", "path": "models/deepfake_best.h5", "kind": "full_model"},
            {"name": "deepfake_artifact", "path": "models/deepfake_artifact.h5", "kind": "full_model"},
            {"name": "deepfake_cnn", "path": "models/deepfake_cnn.h5", "kind": "full_model"},
        ]
        detector.available_model_names = [spec["name"] for spec in detector.available_model_specs]
        detector._model_scan_complete = True
        detector.model_names = ["deepfake_cnn", "deepfake_best", "deepfake_artifact"]
        detector.ensemble_models = [object(), object(), object()]
        detector._ensure_hf_loaded = lambda *args, **kwargs: None
        detector.perform_ela = lambda image: (np.zeros((8, 8, 3), dtype=np.uint8), 0.05)
        detector.perform_fft_analysis = lambda image: (np.zeros((8, 8, 3), dtype=np.uint8), 0.07)
        detector.extract_forensic_features = lambda image: {
            "ela_score": 0.05,
            "fft_score": 0.07,
            "blur_score": 0.1,
            "blockiness_score": 0.12,
            "color_score": 0.08,
            "forensic_score": 0.2,
        }

        def fake_detect_faces(image):
            detector._last_detected_face_count = 1
            detector._last_analyzed_region_count = 1
            detector._last_face_detection_mode = "cascade"
            return [(10, 10, 100, 100)]

        seen_model_names = []

        def fake_predict(_model, _face_crop, model_name):
            seen_model_names.append(model_name)
            return 0.9 if model_name == "deepfake_best" else 0.72

        detector.detect_faces = fake_detect_faces
        detector.predict_with_model = fake_predict

        result = detector.detect_deepfake_ensemble(np.zeros((224, 224, 3), dtype=np.uint8))

        self.assertCountEqual(seen_model_names, ["deepfake_best", "deepfake_artifact"])
        self.assertEqual(result["models_used"], ["deepfake_best", "deepfake_artifact"])

    def test_extract_forensic_features_distinguishes_checkerboard(self):
        detector = self._build_detector()

        natural = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            natural[:, i, :] = i

        checker = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 4):
            for j in range(0, 224, 4):
                checker[i:i + 2, j:j + 2, :] = 255

        natural_features = detector.extract_forensic_features(natural)
        checker_features = detector.extract_forensic_features(checker)

        self.assertGreater(checker_features["forensic_score"], natural_features["forensic_score"])
        self.assertGreater(checker_features["blockiness_score"], natural_features["blockiness_score"])

    def test_aggregate_face_scores_balances_peak_and_consensus(self):
        detector = self._build_detector()
        face_results = [
            {"score": 0.82, "bbox": (0, 0, 120, 120)},
            {"score": 0.78, "bbox": (10, 10, 100, 100)},
            {"score": 0.35, "bbox": (20, 20, 40, 40)},
        ]

        score, consistency = detector._aggregate_face_scores(face_results)

        self.assertGreater(score, 0.65)
        self.assertLess(score, 0.82)
        self.assertGreater(consistency, 0.5)

    def test_detect_faces_preserves_total_detected_count_above_analysis_cap(self):
        detector = self._build_detector()
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        detector.face_cascades = [
            (
                "frontal_default",
                _ScaledCascade(
                    [
                        (10, 10, 80, 80),
                        (110, 10, 80, 80),
                        (210, 10, 80, 80),
                        (10, 110, 80, 80),
                        (110, 110, 80, 80),
                        (210, 110, 80, 80),
                        (10, 210, 80, 80),
                    ],
                    original_width=image.shape[1],
                ),
            )
        ]

        with patch.object(detector, "_detect_faces_mtcnn", return_value=[]), patch.object(
            detector, "_detect_faces_mediapipe", return_value=[]
        ):
            faces = detector.detect_faces(image)

        self.assertEqual(detector._last_face_detection_mode, "cascade")
        self.assertEqual(detector._last_detected_face_count, 7)
        self.assertEqual(detector._last_analyzed_region_count, detector.max_face_regions)
        self.assertEqual(len(faces), detector.max_face_regions)

    def test_detect_faces_keeps_all_five_faces_when_under_limit(self):
        detector = self._build_detector()

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        detector.face_cascades = [
            (
                "frontal_default",
                _ScaledCascade(
                    [
                        (10, 10, 80, 80),
                        (15, 15, 78, 78),
                        (110, 10, 80, 80),
                        (210, 10, 80, 80),
                        (10, 110, 80, 80),
                        (110, 110, 80, 80),
                    ],
                    original_width=image.shape[1],
                ),
            )
        ]

        with patch.object(detector, "_detect_faces_mtcnn", return_value=[]), patch.object(
            detector, "_detect_faces_mediapipe", return_value=[]
        ):
            faces = detector.detect_faces(image)

        self.assertEqual(detector._last_face_detection_mode, "cascade")
        self.assertEqual(detector._last_detected_face_count, 5)
        self.assertEqual(detector._last_analyzed_region_count, 5)
        self.assertEqual(len(faces), 5)

    def test_detect_faces_merges_jittered_overlap_boxes_for_single_face(self):
        detector = self._build_detector()

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        detector.face_cascades = [
            (
                "frontal_default",
                _ScaledCascade(
                    [
                        (20, 20, 90, 90),
                        (24, 26, 88, 88),
                        (72, 22, 78, 78),
                    ],
                    original_width=image.shape[1],
                ),
            )
        ]

        with patch.object(detector, "_detect_faces_mtcnn", return_value=[]), patch.object(
            detector, "_detect_faces_mediapipe", return_value=[]
        ):
            faces = detector.detect_faces(image)

        self.assertEqual(detector._last_face_detection_mode, "cascade")
        self.assertEqual(detector._last_detected_face_count, 1)
        self.assertEqual(detector._last_analyzed_region_count, 1)
        self.assertEqual(len(faces), 1)

    def test_detect_faces_uses_cascade_before_optional_detectors(self):
        detector = self._build_detector()
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.face_cascades = [
            ("frontal_default", _ScaledCascade([(20, 20, 90, 90)], original_width=320))
        ]

        with patch.object(detector, "_detect_faces_mtcnn") as mtcnn_mock, patch.object(
            detector, "_detect_faces_mediapipe"
        ) as mediapipe_mock:
            faces = detector.detect_faces(image)

        mtcnn_mock.assert_not_called()
        mediapipe_mock.assert_not_called()
        self.assertEqual(detector._last_face_detection_mode, "cascade")
        self.assertEqual(faces, [(20, 20, 90, 90)])

    def test_detect_faces_uses_advanced_validator_when_cascade_overcounts(self):
        detector = self._build_detector()
        detector.enable_advanced_face_detectors = True

        image = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.face_cascades = [
            (
                "frontal_default",
                _ScaledCascade(
                    [(20, 20, 90, 90), (160, 25, 82, 82)],
                    original_width=320,
                ),
            )
        ]

        with patch.object(
            detector,
            "_detect_faces_mtcnn",
            return_value=[(24, 24, 88, 88)],
        ) as mtcnn_mock, patch.object(
            detector,
            "_detect_faces_mediapipe",
            return_value=[],
        ) as mediapipe_mock:
            faces = detector.detect_faces(image)

        mtcnn_mock.assert_called_once()
        mediapipe_mock.assert_called_once()
        self.assertEqual(detector._last_face_detection_mode, "mtcnn")
        self.assertEqual(detector._last_detected_face_count, 1)
        self.assertEqual(detector._last_analyzed_region_count, 1)
        self.assertEqual(faces, [(24, 24, 88, 88)])

    def test_advanced_face_selection_prefers_detector_consensus(self):
        detector = self._build_detector()
        detector.enable_advanced_face_detectors = True

        image = np.zeros((320, 320, 3), dtype=np.uint8)

        with patch.object(
            detector,
            "_detect_faces_mtcnn",
            return_value=[(20, 20, 90, 90), (190, 22, 78, 78)],
        ), patch.object(
            detector,
            "_detect_faces_mediapipe",
            return_value=[(24, 24, 88, 88)],
        ):
            mode, faces = detector._select_advanced_face_boxes(image)

        self.assertEqual(mode, "advanced_consensus")
        self.assertEqual(len(faces), 1)

    def test_noisy_cascade_uses_mtcnn_to_stabilize_face_count(self):
        detector = self._build_detector()

        image = np.zeros((480, 480, 3), dtype=np.uint8)
        detector.face_cascades = [
            (
                "frontal_default",
                _ScaledCascade(
                    [
                        (10, 10, 60, 60),
                        (95, 10, 60, 60),
                        (180, 10, 60, 60),
                        (265, 10, 60, 60),
                        (350, 10, 60, 60),
                        (10, 95, 60, 60),
                        (95, 95, 60, 60),
                        (180, 95, 60, 60),
                        (265, 95, 60, 60),
                        (350, 95, 60, 60),
                        (10, 180, 60, 60),
                        (95, 180, 60, 60),
                    ],
                    original_width=image.shape[1],
                ),
            )
        ]

        with patch.object(
            detector,
            "_detect_faces_mtcnn",
            return_value=[
                (20, 20, 90, 90),
                (130, 20, 90, 90),
                (240, 20, 90, 90),
                (75, 145, 90, 90),
                (185, 145, 90, 90),
            ],
        ), patch.object(
            detector,
            "_detect_faces_mediapipe",
            return_value=[],
        ):
            faces = detector.detect_faces(image)

        self.assertEqual(detector._last_face_detection_mode, "mtcnn")
        self.assertEqual(detector._last_detected_face_count, 5)
        self.assertEqual(detector._last_analyzed_region_count, 5)
        self.assertEqual(len(faces), 5)

    def test_forensic_fallback_tracks_regions_without_inflating_face_count(self):
        detector = self._build_detector()
        detector.face_cascades = [("frontal_default", _EmptyCascade())]
        detector._ensure_hf_loaded = lambda *args, **kwargs: None
        detector.perform_ela = lambda image: (np.zeros((8, 8, 3), dtype=np.uint8), 0.05)
        detector.perform_fft_analysis = lambda image: (np.zeros((8, 8, 3), dtype=np.uint8), 0.07)
        detector.extract_forensic_features = lambda image: {
            "ela_score": 0.05,
            "fft_score": 0.07,
            "blur_score": 0.1,
            "blockiness_score": 0.12,
            "color_score": 0.08,
            "forensic_score": 0.2,
        }

        detector._detect_faces_mtcnn = lambda *args, **kwargs: []
        detector._detect_faces_mediapipe = lambda *args, **kwargs: []

        result = detector.detect_deepfake_ensemble(np.zeros((220, 220, 3), dtype=np.uint8))

        self.assertEqual(result["status"], "FORENSIC_ONLY")
        self.assertEqual(result["face_detection_mode"], "fallback_crop")
        self.assertEqual(result["face_count"], 0)
        self.assertEqual(result["analyzed_region_count"], 2)

    def test_ensure_models_ready_skips_pretrained_fallback_by_default(self):
        detector = self._build_detector()

        with patch.object(detector, "_load_existing_models", lambda requested_names=None: None), patch.object(
            detector,
            "_load_pretrained_models",
        ) as pretrained_mock:
            detector._ensure_models_ready()

        pretrained_mock.assert_not_called()
        self.assertEqual(detector.ensemble_models, [])

    def test_detect_faces_uses_mtcnn_when_available(self):
        detector = self._build_detector()
        detector.enable_advanced_face_detectors = True
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.face_cascades = [("frontal_default", _EmptyCascade())]

        with patch.object(
            detector,
            "_detect_faces_mtcnn",
            return_value=[(20, 20, 90, 90), (180, 25, 82, 82)],
        ), patch.object(
            detector,
            "_detect_faces_mediapipe",
            return_value=[],
        ):
            faces = detector.detect_faces(image)

        self.assertEqual(detector._last_face_detection_mode, "mtcnn")
        self.assertEqual(detector._last_detected_face_count, 2)
        self.assertEqual(len(faces), 2)

    def test_profile_cascade_only_runs_when_frontal_search_fails(self):
        detector = self._build_detector()
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.face_cascades = [
            ("frontal_default", _ScaledCascade([(20, 20, 90, 90)], original_width=320)),
            ("profile", _ScaledCascade([(190, 40, 70, 70)], original_width=320)),
        ]

        with patch.object(detector, "_detect_faces_mtcnn", return_value=[]), patch.object(
            detector, "_detect_faces_mediapipe", return_value=[]
        ):
            faces = detector.detect_faces(image)

        self.assertEqual(detector._last_face_detection_mode, "cascade")
        self.assertEqual(detector._last_detected_face_count, 1)
        self.assertEqual(faces, [(20, 20, 90, 90)])

    def test_model_info_exposes_recommended_threshold_hint(self):
        with patch.object(
            DeepfakeDetectorAdvanced,
            "_discover_model_files",
            return_value=["models/deepfake_best_20260315_202503.weights.h5"],
        ), patch.object(
            DeepfakeDetectorAdvanced,
            "_load_evaluation_summary_map",
            return_value={
                "20260316_010648": {
                    "accuracy": 0.91475,
                    "recommended_threshold": 0.44,
                }
            },
        ):
            detector = DeepfakeDetectorAdvanced()
            info = detector.get_model_info()

        self.assertAlmostEqual(info["recommended_threshold"], 0.44, places=2)

    def test_default_threshold_auto_uses_recommended_hint(self):
        with patch.object(
            DeepfakeDetectorAdvanced,
            "_discover_model_files",
            return_value=["models/deepfake_best_20260315_202503.weights.h5"],
        ), patch.object(
            DeepfakeDetectorAdvanced,
            "_load_evaluation_summary_map",
            return_value={
                "20260316_010648": {
                    "accuracy": 0.91475,
                    "recommended_threshold": 0.44,
                }
            },
        ):
            detector = DeepfakeDetectorAdvanced()

        self.assertAlmostEqual(detector.threshold, 0.44, places=2)

    def test_detect_deepfake_video_advanced_uses_auto_sample_interval(self):
        detector = self._build_detector()
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(31)]
        capture = _FakeVideoCapture(frames, fps=30.0)

        def fake_detect(_frame, return_heatmap=False):
            return {
                "is_deepfake": False,
                "confidence": 12.5,
                "ensemble_score": 0.1,
            }

        detector.detect_deepfake_ensemble = fake_detect

        with patch("deepfake_detector_advanced.cv2.VideoCapture", return_value=capture):
            result = detector.detect_deepfake_video_advanced("dummy.mp4")

        self.assertEqual(result["frame_stride"], 10)
        self.assertEqual(result["total_frames_analyzed"], 4)

    def test_detect_with_single_model_returns_visual_artifacts_when_requested(self):
        detector = self._build_detector()
        detector._ensure_hf_loaded = lambda *args, **kwargs: None
        detector.model_names = ["deepfake_demo"]
        detector.ensemble_models = [_FakeGradCamModel()]
        detector.detect_faces = lambda _image: [(16, 16, 80, 80)]
        detector._last_detected_face_count = 1
        detector._last_analyzed_region_count = 1
        detector._last_face_detection_mode = "cascade"
        detector.predict_with_model = lambda _model, _face_crop, _model_name: 0.82
        detector.extract_forensic_features = lambda _image: {
            "ela_score": 0.08,
            "fft_score": 0.12,
            "blur_score": 0.2,
            "blockiness_score": 0.18,
            "color_score": 0.09,
            "forensic_score": 0.3,
        }
        detector.perform_ela = lambda _image: (np.ones((8, 8, 3), dtype=np.uint8), 0.14)
        detector.perform_fft_analysis = lambda _image: (np.ones((8, 8, 3), dtype=np.uint8) * 2, 0.22)
        detector.preprocess_for_model = lambda *_args, **_kwargs: np.zeros((1, 8, 8, 3), dtype=np.float32)
        detector.get_gradcam_heatmap = lambda *_args, **_kwargs: np.ones((8, 8), dtype=np.float32)
        detector.apply_heatmap = lambda *_args, **_kwargs: np.ones((8, 8, 3), dtype=np.uint8) * 127

        result = detector.detect_with_single_model(
            np.zeros((128, 128, 3), dtype=np.uint8),
            "deepfake_demo",
            return_heatmap=True,
        )

        self.assertIsNotNone(result["heatmap"])
        self.assertIsNotNone(result["ela_image"])
        self.assertIsNotNone(result["fft_image"])
        self.assertIn("blockiness_score", result)
        self.assertIn("color_score", result)

    def test_detect_video_with_single_model_adds_representative_frame_artifacts(self):
        detector = self._build_detector()
        frames = [np.full((32, 32, 3), fill_value=value, dtype=np.uint8) for value in (10, 20, 30)]
        capture = _FakeVideoCapture(frames, fps=12.0)
        call_log = []

        def fake_detect(frame, model_name, return_heatmap=False, include_visual_artifacts=True):
            marker = int(frame[0, 0, 0])
            call_log.append((marker, return_heatmap, include_visual_artifacts))
            base = {
                10: {"is_deepfake": False, "confidence": 60.0, "ensemble_score": 0.21},
                20: {"is_deepfake": True, "confidence": 81.0, "ensemble_score": 0.88},
                30: {"is_deepfake": False, "confidence": 58.0, "ensemble_score": 0.34},
            }[marker]
            if include_visual_artifacts:
                base.update(
                    {
                        "heatmap": np.ones((6, 6, 3), dtype=np.uint8),
                        "ela_image": np.ones((6, 6, 3), dtype=np.uint8) * 2,
                        "ela_score": 0.11,
                        "fft_image": np.ones((6, 6, 3), dtype=np.uint8) * 3,
                        "fft_score": 0.19,
                        "target_face_bbox": (4, 4, 20, 20),
                        "face_count": 1,
                        "model_scores": {model_name: base["ensemble_score"]},
                        "forensic_score": 0.27,
                        "blur_score": 0.09,
                        "blockiness_score": 0.13,
                        "color_score": 0.07,
                        "face_detection_mode": "cascade",
                        "analyzed_region_count": 1,
                    }
                )
            return base

        detector.detect_with_single_model = fake_detect

        with patch("deepfake_detector_advanced.cv2.VideoCapture", return_value=capture):
            result = detector.detect_video_with_single_model(
                "dummy.mp4",
                "deepfake_demo",
                sample_rate=1,
                return_heatmap=True,
            )

        self.assertIsNotNone(result["heatmap"])
        self.assertIsNotNone(result["ela_image"])
        self.assertIsNotNone(result["fft_image"])
        self.assertEqual(result["face_count"], 1)
        self.assertEqual(call_log[-1], (20, True, True))
        self.assertEqual(call_log[:-1], [(10, False, False), (20, False, False), (30, False, False)])

    def test_video_runtime_passes_return_heatmap_to_worker(self):
        with patch.object(safe_deepfake_runtime, "_run_worker", return_value={"ok": True}) as worker_mock:
            safe_deepfake_runtime.run_isolated_deepfake_video_analysis(
                "demo.mp4",
                threshold=0.55,
                model_name="deepfake_demo",
                return_heatmap=True,
            )

        payload = worker_mock.call_args.args[0]
        self.assertTrue(payload["return_heatmap"])


if __name__ == "__main__":
    unittest.main()
