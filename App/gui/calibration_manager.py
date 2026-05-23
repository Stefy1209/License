from __future__ import annotations

import threading
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QFileDialog, QMessageBox, QDialog,
    QLineEdit, QTextEdit, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.components import NavBar, StyledButton, StatusBadge
from gui.styles import SUCCESS, WARNING, DANGER


def _rms_variant(rms: float) -> str:
    if rms < 0.5:
        return "success"
    if rms < 1.0:
        return "warning"
    return "danger"


class CalibrationManagerWidget(QWidget):
    """CRUD screen for the calibration library stored in SQLite."""

    back_requested = pyqtSignal()
    _refresh_signal = pyqtSignal()
    _error_signal   = pyqtSignal(str)

    def __init__(self, cfg, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._refresh_signal.connect(self._refresh_list)
        self._error_signal.connect(self._show_error)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        navbar = NavBar("Calibration Library")
        navbar.back_clicked.connect(self.back_requested)

        import_btn = StyledButton("Import .npz", icon_name="document-import")
        import_btn.setFixedWidth(140)
        import_btn.clicked.connect(self._import)
        navbar.add_right_widget(import_btn)
        root.addWidget(navbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._inner = QWidget()
        self._list_layout = QVBoxLayout(self._inner)
        self._list_layout.setContentsMargins(40, 24, 40, 40)
        self._list_layout.setSpacing(12)

        scroll.setWidget(self._inner)
        root.addWidget(scroll, stretch=1)

        self._refresh_list()

    # ── List management ───────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        """Clears and rebuilds the card list from the DB."""
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        from persistence import make_service
        records = make_service().list_all()

        if not records:
            empty = QLabel("No calibrations saved. Run a calibration session or import an .npz file.")
            empty.setObjectName("dim")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._list_layout.addWidget(empty)
        else:
            for rec in records:
                card = _CalibrationCard(rec)
                card.use_clicked.connect(lambda r=rec: self._use(r))
                card.export_clicked.connect(lambda r=rec: self._export(r))
                card.edit_clicked.connect(lambda r=rec: self._edit(r))
                card.delete_clicked.connect(lambda r=rec: self._delete(r))
                self._list_layout.addWidget(card)

        self._list_layout.addStretch()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _use(self, record) -> None:
        """Copies the selected calibration to the active config path."""
        reply = QMessageBox.question(
            self, "Confirm",
            f"Set calibration «{record.name or f'#{record.id}'}» as active?\n"
            f"The current calibration.npz will be overwritten.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        dest = self._cfg.calibration.file

        def _run():
            try:
                from persistence import make_service
                make_service().apply(record.id, dest)
                self._refresh_signal.emit()
            except Exception as exc:
                self._error_signal.emit(str(exc))

        threading.Thread(target=_run, daemon=True, name="cal-apply").start()

    def _export(self, record) -> None:
        default = f"{record.name or f'calibration_{record.id}'}.npz"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export calibration", default, "NumPy archive (*.npz)"
        )
        if not path:
            return

        def _run():
            try:
                from persistence import make_service
                make_service().export(record.id, path)
            except Exception as exc:
                self._error_signal.emit(str(exc))

        threading.Thread(target=_run, daemon=True, name="cal-export").start()

    def _edit(self, record) -> None:
        dialog = _EditDialog(record.name or "", record.notes or "", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, notes = dialog.values()

        def _run():
            try:
                from persistence import make_service
                make_service().update_metadata(record.id, name, notes)
                self._refresh_signal.emit()
            except Exception as exc:
                self._error_signal.emit(str(exc))

        threading.Thread(target=_run, daemon=True, name="cal-edit").start()

    def _delete(self, record) -> None:
        reply = QMessageBox.question(
            self, "Confirm",
            f"Delete calibration «{record.name or f'#{record.id}'}»?\n"
            f"The associated .npz file will be permanently removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        def _run():
            try:
                from persistence import make_service
                make_service().delete(record.id)
                self._refresh_signal.emit()
            except Exception as exc:
                self._error_signal.emit(str(exc))

        threading.Thread(target=_run, daemon=True, name="cal-delete").start()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import calibration file", "", "NumPy archive (*.npz)"
        )
        if not path:
            return
        dialog = _EditDialog("", "", parent=self, title="Import details")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, notes = dialog.values()

        def _run():
            try:
                from persistence import make_service
                make_service().import_file(path, name or None, notes or None)
                self._refresh_signal.emit()
            except Exception as exc:
                self._error_signal.emit(str(exc))

        threading.Thread(target=_run, daemon=True, name="cal-import").start()

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


# ── Card widget ───────────────────────────────────────────────────────────────

class _CalibrationCard(QWidget):
    """Displays one calibration record with action buttons."""

    use_clicked    = pyqtSignal()
    export_clicked = pyqtSignal()
    edit_clicked   = pyqtSignal()
    delete_clicked = pyqtSignal()

    def __init__(self, record, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # Header row: name + RMS badge
        header = QHBoxLayout()
        name_lbl = QLabel(record.name or f"Calibrare #{record.id}")
        name_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        header.addWidget(name_lbl)
        header.addStretch()

        rms_text    = f"RMS {record.rms:.4f} px"
        rms_variant = _rms_variant(record.rms)
        header.addWidget(StatusBadge(rms_text, variant=rms_variant))
        layout.addLayout(header)

        # Info row
        date_str = record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "—"
        board_str = (
            f"{record.cols}×{record.rows} cols/rows, {record.square_mm} mm/square"
            if record.cols else "board params: —"
        )
        info_lbl = QLabel(f"{date_str}  ·  {board_str}")
        info_lbl.setObjectName("dim")
        layout.addWidget(info_lbl)

        if record.notes:
            notes_lbl = QLabel(record.notes)
            notes_lbl.setObjectName("dim")
            notes_lbl.setWordWrap(True)
            layout.addWidget(notes_lbl)

        # Action buttons
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.setContentsMargins(0, 0, 0, 0)

        use_btn    = StyledButton("Use",    icon_name="dialog-ok")
        export_btn = StyledButton("Export", icon_name="document-export")
        edit_btn   = StyledButton("Edit",   icon_name="document-edit")
        delete_btn = StyledButton("Delete", icon_name="edit-delete")

        for btn in (use_btn, export_btn, edit_btn, delete_btn):
            btn.setFixedWidth(100)

        use_btn.clicked.connect(self.use_clicked)
        export_btn.clicked.connect(self.export_clicked)
        edit_btn.clicked.connect(self.edit_clicked)
        delete_btn.clicked.connect(self.delete_clicked)

        actions.addWidget(use_btn)
        actions.addWidget(export_btn)
        actions.addWidget(edit_btn)
        actions.addWidget(delete_btn)
        actions.addStretch()
        layout.addLayout(actions)


# ── Edit / import metadata dialog ─────────────────────────────────────────────

class _EditDialog(QDialog):
    """Simple dialog for editing name and notes of a calibration record."""

    def __init__(
        self,
        name:   str,
        notes:  str,
        parent: QWidget = None,
        title:  str = "Editează calibrarea",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(QLabel("Name (optional):"))
        self._name_edit = QLineEdit(name)
        self._name_edit.setPlaceholderText("e.g. outdoor_daylight_v3")
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Notes (optional):"))
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlainText(notes)
        self._notes_edit.setPlaceholderText("Calibration conditions, camera used, etc.")
        self._notes_edit.setFixedHeight(100)
        layout.addWidget(self._notes_edit)

        btns = QHBoxLayout()
        btns.addStretch()
        ok_btn     = StyledButton("Ok",     icon_name="dialog-ok")
        cancel_btn = StyledButton("Cancel", icon_name="dialog-cancel")
        ok_btn.setFixedWidth(100)
        cancel_btn.setFixedWidth(100)
        ok_btn.clicked.connect(self._accept)
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _accept(self) -> None:
        from persistence.validators.calibration_validator import CalibrationValidator, CalibrationValidationError
        validator = CalibrationValidator()
        try:
            validator.validate_name(self._name_edit.text())
            validator.validate_notes(self._notes_edit.toPlainText())
        except CalibrationValidationError as exc:
            QMessageBox.warning(self, "Validation", str(exc))
            return
        self.accept()

    def values(self) -> tuple[str, str]:
        return self._name_edit.text().strip(), self._notes_edit.toPlainText().strip()
