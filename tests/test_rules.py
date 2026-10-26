"""Deterministic code tests for the parking rule engine."""

from datetime import time
import pytest

from src.schemas import (
    ClearanceVerdict,
    ParkingRule,
    RestrictionType,
    VehicleClass,
)

from src.rules import evaluate_parking_permission


def test_loading_zone_allows_truck_during_hours():
    rule = ParkingRule(
        restriction_type=RestrictionType.LOADING_ZONE,
        start_time=time(7, 0),
        end_time=time(18, 0),
        days=["MON", "TUE", "WED", "THU", "FRI"],
        max_duration_minutes=30,
        commercial_only=True,
    )

    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.TRUCK,
        arrival_time=time(10, 30),
        arrival_day="TUE",
    )

    assert isinstance(verdict, ClearanceVerdict)
    assert verdict.allowed is True
    assert verdict.max_stay_minutes == 30


def test_loading_zone_blocks_passenger_car():
    rule = ParkingRule(
        restriction_type=RestrictionType.LOADING_ZONE,
        start_time=time(7, 0),
        end_time=time(18, 0),
        days=["MON", "TUE", "WED", "THU", "FRI"],
        max_duration_minutes=30,
        commercial_only=True,
    )

    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.SEDAN,
        arrival_time=time(10, 30),
        arrival_day="TUE",
    )

    assert verdict.allowed is False
    assert "commercial" in verdict.reason.lower()


def test_clearway_blocks_all_vehicles_morning_peak():
    rule = ParkingRule(
        restriction_type=RestrictionType.CLEARWAY,
        start_time=time(6, 0),
        end_time=time(10, 0),
        days=["MON", "TUE", "WED", "THU", "FRI"],
        commercial_only=False,
    )

    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.TRUCK,
        arrival_time=time(8, 15),
        arrival_day="MON",
    )

    assert verdict.allowed is False
    assert "clearway" in verdict.reason.lower()