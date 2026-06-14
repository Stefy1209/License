"""Fixtures comune pentru testele CRUD ale librăriei de calibrări.

Fiecare test rulează izolat: bază de date SQLite pe fișier temporar și un director
temporar de fișiere, oferite de fixture-ul `tmp_path` al pytest. DB-ul real
(`calibrations.db`) și directorul `calibration_files` nu sunt atinse niciodată.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Adaugă directorul App/ în sys.path ca `import persistence` să funcționeze
# indiferent din ce director e rulat pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from persistence import Base
from persistence.repository import models  # noqa: F401 — înregistrează modelul ORM pe Base
from persistence.service.calibration_service import CalibrationLibraryService


@pytest.fixture
def session_factory(tmp_path):
    """sessionmaker legat de o bază de date SQLite temporară, cu tabelele create."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture
def session(session_factory):
    """O sesiune deschisă, pentru testele directe pe repository."""
    with session_factory() as s:
        yield s


@pytest.fixture
def files_dir(tmp_path):
    """Director temporar în care service-ul copiază fișierele .npz gestionate."""
    d = tmp_path / "calibration_files"
    d.mkdir()
    return d


@pytest.fixture
def service(session_factory, files_dir):
    """Instanță CalibrationLibraryService legată de DB-ul și directorul temporare."""
    return CalibrationLibraryService(session_factory, str(files_dir))


@pytest.fixture
def sample_npz(tmp_path):
    """Un fișier .npz valid, cu array-urile și scalarii ceruți de validator."""
    path = tmp_path / "calib.npz"
    np.savez(
        path,
        camera_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        rms=np.array(0.5),
        cols=np.array(9),
        rows=np.array(6),
        square_mm=np.array(25.0),
    )
    return str(path)
