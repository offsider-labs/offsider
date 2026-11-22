"""Score image-to-verdict output against manually labelled parking scenarios."""

import argparse
import csv
from datetime import time
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter

import cv2

from src.pipeline import get_detector, process_frame
from src.schemas import VehicleClass


EVAL_DIR = Path(__file__).resolve().parent


def load_scenarios() -> list[dict[str, str]]:
    csv_path = EVAL_DIR / "expected-verdicts.csv"
    with csv_path.open(newline="", encoding="utf-8") as labels:
        return list(csv.DictReader(labels))


def score_scenario(scenario: dict[str, str]) -> dict[str, str | bool | float]:
    """Run the real image pipeline and compare the three labelled verdict fields."""
    if get_detector().use_mock:
        raise RuntimeError("Image scenarios require the real YOLO detector.")

    path = EVAL_DIR / "test-images" / scenario["image"]
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read {path}")

    start = perf_counter()
    verdict, _ = process_frame(
        frame,
        VehicleClass(scenario["vehicle"]),
        arrival_time=time.fromisoformat(scenario["time"]),
        arrival_day=scenario["day"],
    )
    latency_ms = (perf_counter() - start) * 1000.0

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
        "latency_ms": round(latency_ms, 2),
        "reason": verdict.reason,
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[94]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path for the scored CSV.")
    args = parser.parse_args()

    results = [score_scenario(scenario) for scenario in load_scenarios()]
    if not results:
        raise ValueError("No scenarios found in expected verdicts CSV.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    matches = sum(bool(row["exact_match"]) for row in results)
    status_matches = sum(row["actual_status"] == row["expected_status"] for row in results)
    active_matches = sum(row["actual_active"] == row["expected_active"] for row in results)
    stay_matches = sum(
        row["actual_max_stay_minutes"] == row["expected_max_stay_minutes"] for row in results
    )
    latencies = [float(row["latency_ms"]) for row in results]

    print(f"Images: {len({row['image'] for row in results})}")
    print(f"Scenarios: {total}")
    print(f"Exact output: {matches}/{total} ({matches / total:.1%})")
    print(f"Parking status: {status_matches}/{total} ({status_matches / total:.1%})")
    print(f"Active status: {active_matches}/{total} ({active_matches / total:.1%})")
    print(f"Max stay minutes: {stay_matches}/{total} ({stay_matches / total:.1%})")
    print(
        f"Latency (ms): mean={mean(latencies):.1f} | "
        f"median={median(latencies):.1f} | p95={_p95(latencies):.1f}"
    )
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()