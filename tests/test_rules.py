"""Deterministic code tests for the parking rule engine."""

from datetime import time
import pytest

from src.schemas import (
    ClearanceVerdict,
    ParkingRule,
    RestrictionType,
    VehicleClass,
)

from src.rules import evaluate_parking_permission, parse_text_to_rules


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


def test_no_stopping_blocks_vehicle():
    rule = ParkingRule(
        restriction_type=RestrictionType.NO_STOPPING,
        start_time=time(0, 0),
        end_time=time(23, 59),
        days=["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
    )

    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.SEDAN,
        arrival_time=time(14, 0),
        arrival_day="WED",
    )

    assert verdict.allowed is False
    assert "no stopping" in verdict.reason.lower()


def test_loading_zone_allows_delivery_van_15_mins():
    rule = ParkingRule(
        restriction_type=RestrictionType.LOADING_ZONE,
        start_time=time(8, 0),
        end_time=time(18, 0),
        days=["MON", "TUE", "WED", "THU", "FRI"],
        max_duration_minutes=30,
        commercial_only=True,
    )

    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.VAN,
        arrival_time=time(11, 0),
        arrival_day="THU",
    )

    assert verdict.allowed is True
    assert verdict.max_stay_minutes == 15


def test_meter_parking_allows_stay_limit():
    rule = ParkingRule(
        restriction_type=RestrictionType.METER_PARKING,
        start_time=time(9, 0),
        end_time=time(17, 0),
        days=["MON", "TUE", "WED", "THU", "FRI"],
        max_duration_minutes=60,
    )

    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.SEDAN,
        arrival_time=time(12, 0),
        arrival_day="WED",
    )

    assert verdict.allowed is True
    assert verdict.max_stay_minutes == 60


def test_outside_active_hours_parking_allowed():
    rule = ParkingRule(
        restriction_type=RestrictionType.CLEARWAY,
        start_time=time(6, 0),
        end_time=time(10, 0),
        days=["MON", "TUE", "WED", "THU", "FRI"],
    )

    # Arrive at 12:00 PM (outside Clearway hours)
    verdict = evaluate_parking_permission(
        rules=[rule],
        vehicle=VehicleClass.SEDAN,
        arrival_time=time(12, 0),
        arrival_day="MON",
    )

    assert verdict.allowed is True
    assert verdict.restriction_active is False


def test_parse_text_to_rules_supports_nsw_signs():
    rules = parse_text_to_rules("LOADING ZONE\n15 MINUTES\n8:30AM-6PM\nMON-FRI")
    assert len(rules) == 1
    assert rules[0].restriction_type == RestrictionType.LOADING_ZONE
    assert rules[0].max_duration_minutes == 15
    assert rules[0].start_time == time(8, 30)
    assert rules[0].end_time == time(18, 0)
    assert "MON" in rules[0].days and "FRI" in rules[0].days

    rules_2p = parse_text_to_rules("2P METER\n9AM-5PM\nMON-FRI")
    assert len(rules_2p) == 1
    assert rules_2p[0].restriction_type == RestrictionType.METER_PARKING
    assert rules_2p[0].max_duration_minutes == 120

def test_parse_compact_ocr_time_range():
    rules = parse_text_to_rules("2P\n8AM10PM\nMON-FRI")

    assert len(rules) == 1
    assert rules[0].start_time == time(8)
    assert rules[0].end_time == time(22)
    for hour, expected_active in [(7, False), (12, True), (23, False)]:
        verdict = evaluate_parking_permission(rules, VehicleClass.SEDAN, time(hour), "MON")
        assert verdict.restriction_active == expected_active


def test_bus_zone_without_posted_schedule_applies_on_sunday():
    rules = parse_text_to_rules("BUS\nZONE")

    assert rules[0].restriction_type == RestrictionType.BUS_ZONE
    verdict = evaluate_parking_permission(rules, VehicleClass.SEDAN, time(23, 59, 59), "SUN")
    assert not verdict.allowed
    assert verdict.restriction_active
    assert verdict.max_stay_minutes == 0


def test_bus_zone_honours_posted_hours_and_days():
    rules = parse_text_to_rules("BUS ZONE\n8AM-6PM\nMON-FRI")

    for day, hour, allowed in [("MON", 12, False), ("MON", 7, True), ("SUN", 12, True)]:
        verdict = evaluate_parking_permission(rules, VehicleClass.SEDAN, time(hour), day)
        assert verdict.allowed == allowed


@pytest.mark.parametrize(
    "text, allowed, active, max_stay",
    [
        ("NO STOPPING", False, True, 0),
        ("CLEARWAY\nAT ALL TIMES", False, True, 0),
        ("2P\n8AM10PM", True, True, 120),
        ("2P\n8AM10PM\nMON-FRI", True, False, None),
    ],
)
def test_sunday_verdict_uses_posted_days(text, allowed, active, max_stay):
    rules = parse_text_to_rules(text)
    verdict = evaluate_parking_permission(rules, VehicleClass.SEDAN, time(12), "SUN")

    assert verdict.allowed == allowed
    assert verdict.restriction_active == active
    assert verdict.max_stay_minutes == max_stay


def test_abbreviated_friday_end_of_day_range():
    rules = parse_text_to_rules("2P\n8AM-6PM\nMON-FR")

    assert rules[0].days == ["MON", "TUE", "WED", "THU", "FRI"]
    verdict = evaluate_parking_permission(rules, VehicleClass.SEDAN, time(12), "TUE")
    assert verdict.restriction_active
    assert verdict.max_stay_minutes == 120


@pytest.mark.parametrize("days", ["MON-F?", "MON-", "?-FRI"])
def test_unreadable_day_range_does_not_produce_a_rule(days):
    assert parse_text_to_rules(f"2P\n8AM-6PM\n{days}") == []
