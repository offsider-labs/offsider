from .detector import SignDetector
from .ocr import read_signs


class ParkingPipeline:
    def __init__(self, detector: SignDetector | None = None):
        self.detector = detector or SignDetector()

    def process_frame(self, image):
        signs = self.detector.detect_and_crop(image)
        # Next: parse each sign's OCR lines into ParkingRules (src/schemas.py),
        # then check them with rules.evaluate_parking_permission
        return read_signs(signs)
