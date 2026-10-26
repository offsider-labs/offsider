
from datetime import time

from src.schemas import (
    ClearanceVerdict,
    ParkingRule,
    RestrictionType,
    VehicleClass,
)

def evaluate_parking_permission(
        rules: list[ParkingRule],
        vehicle: VehicleClass,
        arrival_time: time,
        arrival_day: str,
) -> ClearanceVerdict:
    """
    Evaluates whether a vehicle is allowed to park based on the provided rules.

    Args:
        rules (list[ParkingRule]): A list of parking rules to evaluate against.
        vehicle (VehicleClass): The class of the vehicle attempting to park.
        arrival_time (time): The time at which the vehicle arrives.
        arrival_day (str): The day of the week on which the vehicle arrives.
    """

    for rule in rules:
        if arrival_day in rule.days:
            if rule.start_time <= arrival_time <= rule.end_time:
                if rule.restriction_type == RestrictionType.LOADING_ZONE:
                    if vehicle == VehicleClass.TRUCK and rule.commercial_only:
                        return ClearanceVerdict(
                            allowed=True,
                            restriction_active=True,
                            reason="Loading zone active for commercial vehicles.",
                            max_stay_minutes=rule.max_duration_minutes,
                        )
                    else:
                        return ClearanceVerdict(
                            allowed=False,
                            restriction_active=True,
                            reason="Loading zone active; non-commercial vehicle.",
                            max_stay_minutes=None
                        )
                elif rule.restriction_type == RestrictionType.CLEARWAY:
                    return ClearanceVerdict(
                        allowed=False,
                        restriction_active=True,
                        reason="Clearway active; no stopping allowed.",
                        max_stay_minutes=None
                    )
    return ClearanceVerdict(
        allowed=True,
        restriction_active=False,
        reason="No active restrictions; parking allowed.",
        max_stay_minutes=None
    )

