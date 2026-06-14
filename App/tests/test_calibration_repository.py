"""Teste CRUD pentru CalibrationRepository (acces direct la baza de date)."""
from __future__ import annotations

from persistence.domain.calibration import CalibrationRecord
from persistence.repository.calibration_repository import CalibrationRepository


def _make_record(name="cam1", notes="prima calibrare") -> CalibrationRecord:
    """Construiește un CalibrationRecord nou (fără id/created_at)."""
    return CalibrationRecord(
        rms=0.5, cols=9, rows=6, square_mm=25.0,
        file_path=f"/fake/{name}.npz", name=name, notes=notes,
    )



def test_save_assigns_id_and_created_at(session):
    repo = CalibrationRepository(session)

    saved = repo.save(_make_record())

    assert saved.id is not None
    assert saved.created_at is not None
    assert saved.name == "cam1"



def test_find_by_id_returns_saved(session):
    repo = CalibrationRepository(session)
    saved = repo.save(_make_record())

    found = repo.find_by_id(saved.id)

    assert found is not None
    assert found.id == saved.id
    assert found.name == "cam1"


def test_find_by_id_returns_none_for_missing(session):
    repo = CalibrationRepository(session)

    assert repo.find_by_id(9999) is None


def test_find_all_returns_every_record(session):
    repo = CalibrationRepository(session)
    repo.save(_make_record(name="cam1"))
    repo.save(_make_record(name="cam2"))

    all_records = repo.find_all()

    assert len(all_records) == 2
    assert {r.name for r in all_records} == {"cam1", "cam2"}



def test_update_changes_only_name_and_notes(session):
    repo = CalibrationRepository(session)
    saved = repo.save(_make_record())

    saved.name = "redenumit"
    saved.notes = "note noi"
    updated = repo.update(saved)

    assert updated is not None
    assert updated.name == "redenumit"
    assert updated.notes == "note noi"
    assert updated.rms == 0.5


def test_update_returns_none_for_missing(session):
    repo = CalibrationRepository(session)
    ghost = _make_record()
    ghost.id = 9999

    assert repo.update(ghost) is None



def test_delete_removes_record(session):
    repo = CalibrationRepository(session)
    saved = repo.save(_make_record())

    assert repo.delete(saved.id) is True
    assert repo.find_by_id(saved.id) is None


def test_delete_returns_false_for_missing(session):
    repo = CalibrationRepository(session)

    assert repo.delete(9999) is False
