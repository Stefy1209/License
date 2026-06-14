"""Teste CRUD pentru CalibrationLibraryService (logica de business + fișiere)."""
from __future__ import annotations

import os

import pytest

from persistence.validators.calibration_validator import CalibrationValidationError



def test_import_file_creates_record_and_copies_file(service, sample_npz, files_dir):
    record = service.import_file(sample_npz, name="importat", notes="din fișier")

    assert record.id is not None
    assert record.name == "importat"
    assert record.rms == 0.5
    assert os.path.exists(record.file_path)
    assert record.file_path.startswith(str(files_dir))


def test_add_from_calibration_creates_record_and_copies_file(service, sample_npz, files_dir):
    record = service.add_from_calibration(
        sample_npz, rms=0.42, cols=9, rows=6, square_mm=25.0, name="sesiune",
    )

    assert record.id is not None
    assert record.rms == 0.42
    assert os.path.exists(record.file_path)
    assert record.file_path.startswith(str(files_dir))



def test_list_all_returns_created_records(service, sample_npz):
    service.import_file(sample_npz, name="a")
    service.import_file(sample_npz, name="b")

    records = service.list_all()

    assert len(records) == 2
    assert {r.name for r in records} == {"a", "b"}



def test_update_metadata_changes_fields(service, sample_npz):
    record = service.import_file(sample_npz, name="vechi", notes="vechi")

    updated = service.update_metadata(record.id, name="nou", notes="note noi")

    assert updated.name == "nou"
    assert updated.notes == "note noi"


def test_update_metadata_raises_for_missing_id(service):
    with pytest.raises(ValueError):
        service.update_metadata(9999, name="x", notes="y")



def test_delete_removes_record_and_file(service, sample_npz):
    record = service.import_file(sample_npz, name="de_șters")
    assert os.path.exists(record.file_path)

    service.delete(record.id)

    assert service.list_all() == []
    assert not os.path.exists(record.file_path)  # fișierul .npz e șters de pe disc


def test_delete_raises_for_missing_id(service):
    with pytest.raises(ValueError):
        service.delete(9999)



def test_import_file_rejects_missing_file(service):
    with pytest.raises(CalibrationValidationError):
        service.import_file("/cale/inexistenta.npz")


def test_add_from_calibration_rejects_high_rms(service, sample_npz):
    with pytest.raises(CalibrationValidationError):
        service.add_from_calibration(
            sample_npz, rms=42.0, cols=9, rows=6, square_mm=25.0,
        )


def test_update_metadata_rejects_too_long_name(service, sample_npz):
    record = service.import_file(sample_npz)

    with pytest.raises(CalibrationValidationError):
        service.update_metadata(record.id, name="x" * 256, notes="")
