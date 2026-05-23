from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from persistence.domain.calibration import CalibrationRecord
from persistence.repository.models import CalibrationModel


class CalibrationRepository:
    """Data-access layer — translates between domain objects and ORM rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, record: CalibrationRecord) -> CalibrationRecord:
        """Inserts a new record and returns it with the generated id/created_at."""
        model = self._to_model(record)
        self._session.add(model)
        self._session.commit()
        self._session.refresh(model)
        return self._to_domain(model)

    def find_by_id(self, id: int) -> Optional[CalibrationRecord]:
        model = self._session.get(CalibrationModel, id)
        return self._to_domain(model) if model else None

    def find_all(self) -> List[CalibrationRecord]:
        """Returns all records ordered newest first."""
        models = (
            self._session.query(CalibrationModel)
            .order_by(CalibrationModel.created_at.desc())
            .all()
        )
        return [self._to_domain(m) for m in models]

    def update(self, record: CalibrationRecord) -> Optional[CalibrationRecord]:
        """Updates only name and notes fields."""
        model = self._session.get(CalibrationModel, record.id)
        if model is None:
            return None
        model.name  = record.name
        model.notes = record.notes
        self._session.commit()
        self._session.refresh(model)
        return self._to_domain(model)

    def delete(self, id: int) -> bool:
        model = self._session.get(CalibrationModel, id)
        if model is None:
            return False
        self._session.delete(model)
        self._session.commit()
        return True

    @staticmethod
    def _to_model(record: CalibrationRecord) -> CalibrationModel:
        return CalibrationModel(
            name      = record.name,
            notes     = record.notes,
            rms       = record.rms,
            cols      = record.cols,
            rows      = record.rows,
            square_mm = record.square_mm,
            file_path = record.file_path,
        )

    @staticmethod
    def _to_domain(model: CalibrationModel) -> CalibrationRecord:
        return CalibrationRecord(
            id         = model.id,
            name       = model.name,
            notes      = model.notes,
            rms        = model.rms,
            cols       = model.cols,
            rows       = model.rows,
            square_mm  = model.square_mm,
            file_path  = model.file_path,
            created_at = model.created_at,
        )
