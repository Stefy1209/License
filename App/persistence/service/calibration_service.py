from __future__ import annotations

import os
import shutil
import uuid
from typing import List, Optional

from persistence.domain.calibration import CalibrationRecord
from persistence.repository.calibration_repository import CalibrationRepository
from persistence.validators.calibration_validator import CalibrationValidator, CalibrationValidationError


class CalibrationLibraryService:
    """Business-logic layer for the calibration library.

    Each public method opens and closes its own DB session so callers
    (including background threads) do not need to manage sessions.
    """

    def __init__(self, session_factory, files_dir: str) -> None:
        self._session_factory = session_factory
        self._files_dir       = files_dir
        self._validator       = CalibrationValidator()

    # ── Write operations ─────────────────────────────────────────────────────

    def add_from_calibration(
        self,
        npz_path:  str,
        rms:       float,
        cols:      int,
        rows:      int,
        square_mm: float,
        name:      Optional[str] = None,
        notes:     Optional[str] = None,
    ) -> CalibrationRecord:
        """Copies an existing .npz into the managed directory and saves metadata.

        Called automatically after a successful calibration session.
        """
        self._validator.validate_rms(rms)
        self._validator.validate_npz_file(npz_path)
        self._validator.validate_name(name or "")
        self._validator.validate_notes(notes or "")

        dest = self._unique_dest()
        shutil.copy2(npz_path, dest)

        record = CalibrationRecord(
            rms=rms, cols=cols, rows=rows, square_mm=square_mm,
            file_path=dest, name=name, notes=notes,
        )
        with self._session_factory() as session:
            return CalibrationRepository(session).save(record)

    def import_file(
        self,
        npz_path: str,
        name:     Optional[str] = None,
        notes:    Optional[str] = None,
    ) -> CalibrationRecord:
        """Imports an external .npz file chosen by the user.

        Reads rms from the file if present, otherwise defaults to 0.0.
        cols/rows/square_mm default to 0 when not stored in the file.
        """
        self._validator.validate_name(name or "")
        self._validator.validate_notes(notes or "")
        data = self._validator.validate_npz_file(npz_path)

        rms       = float(data["rms"])       if "rms"       in data else 0.0
        cols      = int(data["cols"])        if "cols"       in data else 0
        rows      = int(data["rows"])        if "rows"       in data else 0
        square_mm = float(data["square_mm"]) if "square_mm"  in data else 0.0

        self._validator.validate_rms(rms)

        dest = self._unique_dest()
        shutil.copy2(npz_path, dest)

        record = CalibrationRecord(
            rms=rms, cols=cols, rows=rows, square_mm=square_mm,
            file_path=dest, name=name, notes=notes,
        )
        with self._session_factory() as session:
            return CalibrationRepository(session).save(record)

    def update_metadata(self, id: int, name: str, notes: str) -> CalibrationRecord:
        """Updates name and notes for an existing record."""
        self._validator.validate_name(name)
        self._validator.validate_notes(notes)
        with self._session_factory() as session:
            repo   = CalibrationRepository(session)
            record = repo.find_by_id(id)
            if record is None:
                raise ValueError(f"Calibration with id={id} not found.")
            record.name  = name or None
            record.notes = notes or None
            return repo.update(record)

    def delete(self, id: int) -> None:
        """Removes the DB record and the managed .npz file from disk."""
        with self._session_factory() as session:
            repo   = CalibrationRepository(session)
            record = repo.find_by_id(id)
            if record is None:
                raise ValueError(f"Calibration with id={id} not found.")
            repo.delete(id)
        if record.file_path and os.path.exists(record.file_path):
            os.remove(record.file_path)

    # ── Read / export operations ──────────────────────────────────────────────

    def list_all(self) -> List[CalibrationRecord]:
        with self._session_factory() as session:
            return CalibrationRepository(session).find_all()

    def apply(self, id: int, dest_path: str) -> None:
        """Copies the managed .npz to dest_path (the path used by the pipeline)."""
        with self._session_factory() as session:
            record = CalibrationRepository(session).find_by_id(id)
        if record is None:
            raise ValueError(f"Calibration with id={id} not found.")
        if not os.path.exists(record.file_path):
            raise FileNotFoundError(f"Managed file missing: {record.file_path}")
        shutil.copy2(record.file_path, dest_path)

    def export(self, id: int, export_path: str) -> None:
        """Copies the managed .npz to a user-chosen export path."""
        with self._session_factory() as session:
            record = CalibrationRepository(session).find_by_id(id)
        if record is None:
            raise ValueError(f"Calibration with id={id} not found.")
        if not os.path.exists(record.file_path):
            raise FileNotFoundError(f"Managed file missing: {record.file_path}")
        shutil.copy2(record.file_path, export_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _unique_dest(self) -> str:
        return os.path.join(self._files_dir, f"{uuid.uuid4().hex}.npz")
