"""Pipeline toolkit: panel localization and cropping."""

from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class SignDetector:
    def __init__(
        self,
        weights_path: str | None = "models/best.pt",
        conf_threshold: float = 0.4,
        use_mock: bool = False,
        mock_dir: str = "fixtures/sample_crops",
    ):
        self.conf_threshold = conf_threshold
        self.model = None
        self.use_mock = use_mock
        self.mock_dir = mock_dir

        if not self.use_mock:
            if weights_path and Path(weights_path).exists() and YOLO is not None:
                self.model = YOLO(weights_path)
            else:
                print("Warning: YOLO model not loaded. Using mock crops instead.")
                self.use_mock = True

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads an image into a standard BGR NumPy array."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return img

    def detect_and_crop(
        self,
        image_input: str | np.ndarray,
        mock_crops_dir: str | None = None,
    ) -> list[tuple[np.ndarray, dict]]:
        """
        Locates parking sign panels and returns a list of (crop_array, metadata) tuples.

        crop_array: np.ndarray (H, W, C) in BGR format ready for EasyOCR.
        metadata: dict containing confidence and bounding box coordinates.
        """
        # Mock Mode (if YOLO model is faulty or unavailable)
        if self.use_mock or self.model is None:
            target_dir = mock_crops_dir or self.mock_dir
            return self._load_mock_crops(target_dir)

        # Ensure image is in NumPy array form
        if isinstance(image_input, str):
            img = self.load_image(image_input)
        else:
            img = image_input

        results = self.model.predict(img, conf=self.conf_threshold, verbose=False)
        crops: list[tuple[np.ndarray, dict]] = []

        if not results or len(results[0].boxes) == 0:
            return crops

        h, w, _ = img.shape
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = xyxy.astype(int)

            # Defensive clamp to ensure coordinates stay inside frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # NumPy slice: [rows, cols] -> [y1:y2, x1:x2]
            cropped_panel = img[y1:y2, x1:x2]

            if cropped_panel.size > 0:
                crops.append((
                    cropped_panel,
                    {
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": cls_id,
                    },
                ))

        return crops

    def _load_mock_crops(self, mock_dir: str | None) -> list[tuple[np.ndarray, dict]]:
        """Loads static fixture crops when YOLO model is unavailable or in mock mode."""
        if not mock_dir or not Path(mock_dir).exists():
            return []

        mock_crops: list[tuple[np.ndarray, dict]] = []
        path_obj = Path(mock_dir)
        image_files = sorted(path_obj.glob("*.jpg")) + sorted(path_obj.glob("*.png"))

        for file in image_files:
            crop = cv2.imread(str(file))
            if crop is not None:
                h, w = crop.shape[:2]
                mock_crops.append((
                    crop,
                    {
                        "confidence": 1.0,
                        "bbox": [0, 0, int(w), int(h)],
                        "mock": True,
                    },
                ))
        return mock_crops