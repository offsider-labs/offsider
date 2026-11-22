"""Score image-to-verdict output against manually labelled parking scenarios."""

import argparse
import csv
from datetime import time
from pathlib import Path

import cv2

from src.pipeline import get_detector, process_frame
from src.schemas import VehicleClass


EVAL_DIR = Path(__file__).resolve().parent


def load_scenarios() -> list[dict[str, str]]:
    with (EVAL_DIR / "expected-verdicts.csv").open(newline="", encoding="utf-8") as labels:
        return list(csv.DictReader(labels))


def score_scenario(scenario: dict[str, str]) -> dict[str, str | bool]:
    """Run the real image pipeline and compare the three labelled verdict fields."""
    if get_detector().use_mock:
        raise RuntimeError("Image scenarios require the real YOLO detector.")

    path = EVAL_DIR / "test-images" / scenario["image"]
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read {path}")
    verdict, _ = process_frame(
        frame,
        VehicleClass(scenario["vehicle"]),
        arrival_time=time.fromisoformat(scenario["time"]),
        arrival_day=scenario["day"],
    )
    status = "Check Sign" if verdict.warning else "Allowed" if verdict.allowed else "Restricted"
    actual = {
        "status": status,
        "active": str(verdict.restriction_active).lower(),
        "max_stay_minutes": (
            "" if verdict.max_stay_minutes is None else str(verdict.max_stay_minutes)
        ),
    }
    mismatches = [field for field, value in actual.items() if value != scenario[f"expected_{field}"]]
    return {
        **scenario,
        **{f"actual_{field}": value for field, value in actual.items()},
        "exact_match": not mismatches,
        "mismatched_fields": ", ".join(mismatches),
        "reason": verdict.reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path for the scored CSV.")
    args = parser.parse_args()

    results = [score_scenario(scenario) for scenario in load_scenarios()]

    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    matches = sum(row["exact_match"] for row in results)
    status_matches = sum(row["actual_status"] == row["expected_status"] for row in results)
    active_matches = sum(row["actual_active"] == row["expected_active"] for row in results)
    stay_matches = sum(row["actual_max_stay_minutes"] == row["expected_max_stay_minutes"] for row in results)
    
    print(f"Images: {len({row['image'] for row in results})}")
    print(f"Exact output: {matches}/{len(results)} ({matches / len(results):.1%})")
    print(f"Parking status: {status_matches}/{len(results)} ({status_matches / len(results):.1%})")
    print(f"Active status: {active_matches}/{len(results)} ({active_matches / len(results):.1%})")
    print(f"Max stay minutes: {stay_matches}/{len(results)} ({stay_matches / len(results):.1%})")

    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
