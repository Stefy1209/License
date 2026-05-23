from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

_APP_DIR = Path(__file__).parent.parent

DB_PATH   = _APP_DIR / "calibrations.db"
FILES_DIR = _APP_DIR / "calibration_files"


class Base(DeclarativeBase):
    pass


engine       = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Creates tables and managed-files directory if they don't exist yet."""
    from persistence.repository import models as _  # noqa: F401 — registers ORM models with Base
    FILES_DIR.mkdir(exist_ok=True)
    Base.metadata.create_all(engine)


def make_service():
    """Returns a ready-to-use CalibrationLibraryService."""
    from persistence.service.calibration_service import CalibrationLibraryService
    return CalibrationLibraryService(SessionLocal, str(FILES_DIR))
