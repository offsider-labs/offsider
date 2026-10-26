from .detector import SignDetector
from .ocr import read_signs
from .parser import parse_restrictions
from .rules import evaluate_rules


class ParkingPipeline:
    def __init__(self, detector: SignDetector | None = None):
        self.detector = detector or SignDetector()

    def process_frame(self, image):
        signs = self.detector.detect_and_crop(image)
        texts = read_signs(signs)
        restrictions = parse_restrictions(texts)
        return evaluate_rules(restrictions)
