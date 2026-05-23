from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CalibrationRecord:
    """Pure domain object — no ORM or numpy dependency."""
    rms:       float
    cols:      int
    rows:      int
    square_mm: float
    file_path: str
    id:         Optional[int]      = None
    name:       Optional[str]      = None
    notes:      Optional[str]      = None
    created_at: Optional[datetime] = field(default=None)
