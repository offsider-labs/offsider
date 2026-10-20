from .detector import detect_signs
from .ocr import read_signs
from .parser import parse_restrictions
from .rules import evaluate_rules


class ParkingPipeline:
    def process_frame(self, image):
        signs = detect_signs(image)
        texts = read_signs(signs)
        restrictions = parse_restrictions(texts)
        return evaluate_rules(restrictions)