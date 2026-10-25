from datetime import time
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class VehicleClass(str, Enum):
    TRUCK = "truck"
    VAN = "van"
    SEDAN = "sedan"
    MOTORCYCLE = "motorcycle"


class RestrictionType(str, Enum):
    LOADING_ZONE = "LOADING_ZONE"
    METER_PARKING = "METER"
    TICKET_PARKING = "TICKET"
    CLEARWAY = "CLEARWAY"
    NO_STOPPING = "NO_STOPPING"
    PERMIT_ZONE = "PERMIT"


class ParkingRule(BaseModel):
    """
    Structured representation of a single regulatory sign panel.
    """
    restriction_type: RestrictionType
    start_time: time                     # e.g., time(7, 0) for 07:00
    end_time: time                       # e.g., time(18, 0) for 18:00
    days: List[str] = Field(
        default=["MON", "TUE", "WED", "THU", "FRI"],
        description="Active days: MON, TUE, WED, THU, FRI, SAT, SUN"
    )
    max_duration_minutes: Optional[int] = Field(
        default=None,
        description="Time limit in minutes, e.g., 30 for 1/2P, 60 for 1P"
    )
    commercial_only: bool = Field(
        default=False,
        description="True for Loading Zones or goods-only restrictions"
    )


class ClearanceVerdict(BaseModel):
    """
    The output of the Rule Parser.
    """
    allowed: bool
    restriction_active: bool
    reason: str
    max_stay_minutes: Optional[int] = None
    warning: Optional[str] = None