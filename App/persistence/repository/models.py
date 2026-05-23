from __future__ import annotations

from sqlalchemy import Column, Integer, Float, String, Text, DateTime
from sqlalchemy.sql import func

from persistence import Base


class CalibrationModel(Base):
    __tablename__ = "calibrations"

    id         = Column(Integer,     primary_key=True, autoincrement=True)
    name       = Column(String(255), nullable=True)
    notes      = Column(Text,        nullable=True)
    rms        = Column(Float,       nullable=False)
    cols       = Column(Integer,     nullable=False)
    rows       = Column(Integer,     nullable=False)
    square_mm  = Column(Float,       nullable=False)
    file_path  = Column(String(512), nullable=False, unique=True)
    created_at = Column(DateTime,    server_default=func.now())
