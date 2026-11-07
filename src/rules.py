""" Rules interface: converting raw OCR text to structured rules and evaluating parking clearance. """
from datetime import time
import re
from typing import List, Optional
from src.schemas import (
    ParkingRule,
    ClearanceVerdict,
    RestrictionType,
    VehicleClass,
)

DAY_MAP = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def parse_time_str(time_str: str) -> time:
    """Parses various time formats into datetime.time."""
    clean = time_str.strip().upper().replace(".", ":")
    match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)$", clean)
    if not match:
        raise ValueError(f"Unrecognized time format: {time_str}")

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = match.group(3)

    if meridiem == "PM" and hour != 12:
        hour += 12
    elif meridiem == "AM" and hour == 12:
        hour = 0

    return time(hour, minute)


def parse_days(day_str: str) -> List[str]:
    """Parses 'MON-FRI' into ['MON', 'TUE', 'WED', 'THU', 'FRI'] or single days."""
    clean = day_str.strip().upper()
    range_match = re.search(r"\b(MON|TUE|WED|THU|FRI|SAT|SUN)\s*-\s*(MON|TUE|WED|THU|FRI|SAT|SUN)\b", clean)
    if range_match:
        start_day, end_day = range_match.group(1), range_match.group(2)
        start_idx = DAY_MAP.index(start_day)
        end_idx = DAY_MAP.index(end_day)
        if start_idx <= end_idx:
            return DAY_MAP[start_idx : end_idx + 1]
        return DAY_MAP[start_idx:] + DAY_MAP[: end_idx + 1]

    found = [d for d in DAY_MAP if re.search(rf"\b{d}\b", clean)]
    return found if found else ["MON", "TUE", "WED", "THU", "FRI"]


def parse_text_to_rules(raw_text: str) -> List[ParkingRule]:
    """
    Parses OCR text into ParkingRule objects under NSW parking sign conventions.
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.upper().replace("\n", " ")
    rules: List[ParkingRule] = []

    res_type = None
    commercial_only = False
    duration_mins = None

    if "CLEARWAY" in text:
        res_type = RestrictionType.CLEARWAY
    elif "LOADING ZONE" in text or "LOADING" in text:
        res_type = RestrictionType.LOADING_ZONE
        commercial_only = True
        duration_mins = 30  # standard default for NSW Loading Zones unless specified
        min_match = re.search(r"\b(\d{1,2})\s*(?:MIN|MINS|MINUTES)\b", text)
        if min_match:
            duration_mins = int(min_match.group(1))
    elif "NO STOPPING" in text:
        res_type = RestrictionType.NO_STOPPING
    elif "NO PARKING" in text:
        res_type = RestrictionType.NO_PARKING
        duration_mins = 2  # NSW Rule 168: 2-minute max drop-off
    else:
        # check for duration limits: 1/4P, 1/2P, 1P, 2P, 3P, 4P, etc.
        p_match = re.search(r"\b(1/4|1/2|1|2|3|4|5|6|7|8)\s*P\b", text)
        if p_match:
            val = p_match.group(1)
            if val == "1/4":
                duration_mins = 15
            elif val == "1/2":
                duration_mins = 30
            else:
                duration_mins = int(val) * 60

            if "METER" in text:
                res_type = RestrictionType.METER_PARKING
            elif "TICKET" in text:
                res_type = RestrictionType.TICKET_PARKING
            else:
                res_type = RestrictionType.PERMIT_ZONE
        elif re.search(r"\bP\b", text):
            min_match = re.search(r"\b(\d{1,2})\s*(?:MIN|MINS|MINUTES)\b", text)
            if min_match:
                duration_mins = int(min_match.group(1))
            else:
                duration_mins = 60
            res_type = RestrictionType.PERMIT_ZONE

    if not res_type:
        return rules

    # extract operating hours: e.g., '7:30AM - 6PM' or '7AM - 9AM'
    time_match = re.search(
        r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*-\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM))",
        text
    )

    if time_match:
        try:
            start_t = parse_time_str(time_match.group(1))
            end_t = parse_time_str(time_match.group(2))
        except ValueError:
            start_t, end_t = time(0, 0), time(23, 59)
    else:
        # default full day if no window is listed
        start_t, end_t = time(0, 0), time(23, 59)

    # extract active days
    days = parse_days(text)

    rules.append(
        ParkingRule(
            restriction_type=res_type,
            start_time=start_t,
            end_time=end_t,
            days=days,
            max_duration_minutes=duration_mins,
            commercial_only=commercial_only,
        )
    )

    return rules


def evaluate_parking_permission(
    rules: list[ParkingRule],
    vehicle: VehicleClass,
    arrival_time: time,
    arrival_day: str,
) -> ClearanceVerdict:
    """
    Evaluates whether a vehicle is allowed to park based on the provided rules.
    Follows NSW kerbside compliance law:
    - Clearway and No Stopping strictly prohibit stopping during active hours.
    - No Parking allows immediate 2-minute passenger/goods drop-off only.
    - Loading Zone allows commercial trucks (30 min) and delivery vans (15 min).
    - Timed / Meter / Ticket parking allows parking up to the specified limit.
    - Unrestricted parking when no active restrictions apply.

    Args:
        rules (list[ParkingRule]): A list of parking rules to evaluate against.
        vehicle (VehicleClass): The class of the vehicle attempting to park.
        arrival_time (time): The time at which the vehicle arrives.
        arrival_day (str): The day of the week on which the vehicle arrives.
    """
    active_permissive: list[ClearanceVerdict] = []

    for rule in rules:
        if arrival_day in rule.days:
            if rule.start_time <= arrival_time <= rule.end_time:
                if rule.restriction_type == RestrictionType.CLEARWAY:
                    return ClearanceVerdict(
                        allowed=False,
                        restriction_active=True,
                        reason="Clearway active; no stopping allowed.",
                        max_stay_minutes=0,
                        rules=rules,
                    )
                elif rule.restriction_type == RestrictionType.NO_STOPPING:
                    return ClearanceVerdict(
                        allowed=False,
                        restriction_active=True,
                        reason="No Stopping active; stopping prohibited.",
                        max_stay_minutes=0,
                        rules=rules,
                    )
                elif rule.restriction_type == RestrictionType.NO_PARKING:
                    return ClearanceVerdict(
                        allowed=False,
                        restriction_active=True,
                        reason="No Parking active; 2-minute drop-off only (driver within 3m).",
                        max_stay_minutes=2,
                        rules=rules,
                    )
                elif rule.restriction_type == RestrictionType.LOADING_ZONE:
                    if vehicle == VehicleClass.TRUCK and rule.commercial_only:
                        active_permissive.append(
                            ClearanceVerdict(
                                allowed=True,
                                restriction_active=True,
                                reason="Loading zone active for commercial vehicles.",
                                max_stay_minutes=rule.max_duration_minutes or 30,
                                rules=rules,
                            )
                        )
                    elif vehicle == VehicleClass.VAN and rule.commercial_only:
                        stay = min(rule.max_duration_minutes or 30, 15)
                        active_permissive.append(
                            ClearanceVerdict(
                                allowed=True,
                                restriction_active=True,
                                reason="Loading zone active; delivery van permitted (15 min max).",
                                max_stay_minutes=stay,
                                rules=rules,
                            )
                        )
                    else:
                        return ClearanceVerdict(
                            allowed=False,
                            restriction_active=True,
                            reason="Loading zone active; non-commercial vehicle.",
                            max_stay_minutes=None,
                            rules=rules,
                        )
                elif rule.restriction_type in (
                    RestrictionType.METER_PARKING,
                    RestrictionType.TICKET_PARKING,
                    RestrictionType.PERMIT_ZONE,
                ):
                    p_type = (
                        "Meter"
                        if rule.restriction_type == RestrictionType.METER_PARKING
                        else ("Ticket" if rule.restriction_type == RestrictionType.TICKET_PARKING else "Timed")
                    )
                    limit = rule.max_duration_minutes or 60
                    active_permissive.append(
                        ClearanceVerdict(
                            allowed=True,
                            restriction_active=True,
                            reason=f"{p_type} parking active: parking permitted up to {limit} minutes.",
                            max_stay_minutes=limit,
                            rules=rules,
                        )
                    )

    if active_permissive:
        # Return the most restrictive active permissive rule (shortest stay limit)
        active_permissive.sort(key=lambda v: v.max_stay_minutes if v.max_stay_minutes is not None else 9999)
        return active_permissive[0]

    return ClearanceVerdict(
        allowed=True,
        restriction_active=False,
        reason="No active restrictions; parking allowed.",
        max_stay_minutes=None,
        rules=rules,
    )

