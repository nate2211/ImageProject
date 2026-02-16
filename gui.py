import sys
import traceback
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QIcon, QWheelEvent, QPainter, QColor, QPalette, QBrush
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QLabel, QFileDialog, QSplitter,
    QScrollArea, QComboBox, QGroupBox, QCheckBox, QToolBar, QMessageBox,
    QAbstractItemView, QAction, QListWidgetItem, QLineEdit, QSlider,
    QDoubleSpinBox, QSpinBox, QFormLayout, QFrame, QStyleFactory, QSizePolicy
)

from PIL import Image
import numpy as np

# --- Import your pipeline registry ---
import palettes
import content
import filters
import learning
from palettes import REGISTRY
import os
import sys

# Add the frozen bundle directory to PATH so subprocess can find ffmpeg.exe
if getattr(sys, 'frozen', False):
    os.environ["PATH"] += os.pathsep + sys._MEIPASS

# -----------------------------------------------------------------------------
# Dark Theme & Styling
# -----------------------------------------------------------------------------
def set_dark_theme(app):
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    app.setStyleSheet("""
        QGroupBox { 
            border: 1px solid #555; 
            margin-top: 1.2em; 
            border-radius: 4px; 
            font-weight: bold;
        }
        QGroupBox::title { 
            subcontrol-origin: margin; 
            subcontrol-position: top left; 
            padding: 0 5px; 
        }
        QSlider::groove:horizontal {
            border: 1px solid #3A3939;
            height: 8px;
            background: #201F1F;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #3a8ee6;
            border: 1px solid #3a8ee6;
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px;
            background: #252525;
            selection-background-color: #3a8ee6;
            color: white;
        }
        QListWidget {
            background-color: #252525;
            border: 1px solid #555;
            color: white;
        }
        QListWidget::item:selected {
            background-color: #3a8ee6;
        }
    """)


# -----------------------------------------------------------------------------
# Worker Thread
# -----------------------------------------------------------------------------
class ProcessingWorker(QThread):
    finished = pyqtSignal(object, float)
    error = pyqtSignal(str)

    def __init__(self, input_image: Image.Image, pipeline_specs: List[Dict[str, Any]], global_seed: Optional[int]):
        super().__init__()
        self.input_image = input_image
        self.pipeline_specs = pipeline_specs
        self.seed = global_seed

    def run(self):
        try:
            t0 = time.perf_counter()
            img = self.input_image.copy()
            for stage in self.pipeline_specs:
                name = stage['name']
                params = stage['params']
                gen = REGISTRY.create(name, seed=self.seed)
                img = gen.generate(img, **params)
            t1 = time.perf_counter()
            self.finished.emit(img, t1 - t0)
        except Exception as e:
            tb = "".join(traceback.format_exception(None, e, e.__traceback__))
            stage_name = name if 'name' in locals() else "Unknown"
            self.error.emit(f"Error in stage '{stage_name}':\n{str(e)}\n\nTraceback:\n{tb}")


# -----------------------------------------------------------------------------
# Image Viewer
# -----------------------------------------------------------------------------
class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._drag_start: Optional[QPoint] = None
        self._offset = QPoint(0, 0)
        self.setBackgroundRole(QPalette.Dark)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)

    def set_image(self, pil_image: Image.Image):
        if pil_image is None:
            self.pixmap = None
            self.update()
            return

        # Ensure RGBA format and use Format_RGBA8888 to avoid Blue/Red swap
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")

        data = pil_image.tobytes("raw", "RGBA")
        qim = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
        self.pixmap = QPixmap.fromImage(qim)

        self.fit_to_window()
        self.update()

    def fit_to_window(self):
        if not self.pixmap: return
        ratio_w = self.width() / self.pixmap.width()
        ratio_h = self.height() / self.pixmap.height()
        self._zoom = min(ratio_w, ratio_h) * 0.95
        self.center_image()

    def center_image(self):
        if not self.pixmap: return
        img_w = self.pixmap.width() * self._zoom
        img_h = self.pixmap.height() * self._zoom
        x = (self.width() - img_w) / 2
        y = (self.height() - img_h) / 2
        self._offset = QPoint(int(x), int(y))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        # Checkered background pattern
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.pixmap:
            w = int(self.pixmap.width() * self._zoom)
            h = int(self.pixmap.height() * self._zoom)
            painter.drawPixmap(self._offset.x(), self._offset.y(), w, h, self.pixmap)
        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image Loaded")

    def wheelEvent(self, event: QWheelEvent):
        if not self.pixmap: return
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        old_zoom = self._zoom
        self._zoom = max(0.01, min(50.0, self._zoom * zoom_factor))

        mouse_pos = event.pos()
        vec_x = mouse_pos.x() - self._offset.x()
        vec_y = mouse_pos.y() - self._offset.y()
        new_vec_x = vec_x * (self._zoom / old_zoom)
        new_vec_y = vec_y * (self._zoom / old_zoom)
        self._offset.setX(int(mouse_pos.x() - new_vec_x))
        self._offset.setY(int(mouse_pos.y() - new_vec_y))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_start:
            delta = event.pos() - self._drag_start
            self._offset += delta
            self._drag_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag_start = None


# -----------------------------------------------------------------------------
# Parameter Widget Components
# -----------------------------------------------------------------------------
class FloatParam(QWidget):
    def __init__(self, name, val, min_val, max_val, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)
        self.layout.setSpacing(2)

        # Label Row
        lbl_row = QHBoxLayout()
        self.lbl = QLabel(name)
        lbl_row.addWidget(self.lbl)
        self.layout.addLayout(lbl_row)

        # Control Row
        ctrl_row = QHBoxLayout()

        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setSingleStep((max_val - min_val) / 100.0)
        self.spin.setDecimals(2)
        self.spin.setValue(float(val))
        self.spin.setFixedWidth(70)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.norm_val = (float(val) - min_val) / (max_val - min_val if max_val > min_val else 1.0)
        self.slider.setValue(int(self.norm_val * 1000))

        # Connections
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)

        ctrl_row.addWidget(self.slider)
        ctrl_row.addWidget(self.spin)
        self.layout.addLayout(ctrl_row)

        self.min_val = min_val
        self.max_val = max_val
        self.block_updates = False

    def _spin_changed(self, val):
        if self.block_updates: return
        self.block_updates = True
        ratio = (val - self.min_val) / (self.max_val - self.min_val if self.max_val > self.min_val else 1.0)
        self.slider.setValue(int(ratio * 1000))
        self.block_updates = False

    def _slider_changed(self, val):
        if self.block_updates: return
        self.block_updates = True
        float_val = self.min_val + (val / 1000.0) * (self.max_val - self.min_val)
        self.spin.setValue(float_val)
        self.block_updates = False

    def value(self):
        return self.spin.value()


class IntParam(QWidget):
    def __init__(self, name, val, min_val, max_val, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)
        self.layout.setSpacing(2)

        lbl_row = QHBoxLayout()
        self.lbl = QLabel(name)
        lbl_row.addWidget(self.lbl)
        self.layout.addLayout(lbl_row)

        ctrl_row = QHBoxLayout()

        self.spin = QSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(int(val))
        self.spin.setFixedWidth(70)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(int(val))

        self.block_updates = False
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)

        ctrl_row.addWidget(self.slider)
        ctrl_row.addWidget(self.spin)
        self.layout.addLayout(ctrl_row)

    def _spin_changed(self, val):
        if self.block_updates: return
        self.block_updates = True
        self.slider.setValue(val)
        self.block_updates = False

    def _slider_changed(self, val):
        if self.block_updates: return
        self.block_updates = True
        self.spin.setValue(val)
        self.block_updates = False

    def value(self):
        return self.spin.value()


# -----------------------------------------------------------------------------
# Dynamic Settings Widget
# -----------------------------------------------------------------------------
class StageSettingsWidget(QWidget):
    paramChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        self.controls = {}
        self.updating = False

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.controls.clear()

    def build_ui(self, stage_name: str, current_params: Dict[str, Any]):
        self.clear()
        self.updating = True

        try:
            gen_class = REGISTRY._by_name.get(stage_name)
            if not gen_class:
                self.layout.addWidget(QLabel(f"Unknown generator: {stage_name}"))
                return
            param_defs = gen_class.get_params()
        except Exception as e:
            self.layout.addWidget(QLabel(f"Error loading params: {e}"))
            return

        if not param_defs:
            lbl = QLabel("No configurable parameters.")
            lbl.setStyleSheet("color: gray; font-style: italic;")
            self.layout.addWidget(lbl)
            return

        for p in param_defs:
            name = p['name']
            ptype = p.get('type', str)
            default = p.get('default')
            help_text = p.get('help', '')
            val = current_params.get(name, default)

            container = None

            # 1. Choice
            if 'choices' in p:
                container = QWidget()
                l = QHBoxLayout(container)
                l.setContentsMargins(0, 5, 0, 5)
                lbl = QLabel(name)
                cmb = QComboBox()
                cmb.addItems([str(c) for c in p['choices']])
                idx = cmb.findText(str(val))
                if idx >= 0: cmb.setCurrentIndex(idx)
                cmb.currentTextChanged.connect(self._on_change)
                l.addWidget(lbl)
                l.addWidget(cmb, 1)  # stretch

                container.widget = cmb

            # 2. Float Range
            elif ptype is float and 'min' in p and 'max' in p:
                container = FloatParam(name, val, p['min'], p['max'])
                container.spin.valueChanged.connect(self._on_change)
                container.widget = container  # self-reference for value() call

            # 3. Int Range
            elif ptype is int and 'min' in p and 'max' in p:
                container = IntParam(name, val, p['min'], p['max'])
                container.spin.valueChanged.connect(self._on_change)
                container.widget = container

            # 4. Boolean
            elif ptype is bool:
                container = QCheckBox(name)
                container.setChecked(bool(val))
                container.toggled.connect(self._on_change)
                container.widget = container

            # 5. File/String
            else:
                container = QWidget()
                l = QHBoxLayout(container)
                l.setContentsMargins(0, 5, 0, 5)
                lbl = QLabel(name)
                line = QLineEdit(str(val))
                line.textChanged.connect(self._on_change)
                l.addWidget(lbl)
                l.addWidget(line)

                if name in ('profile', 'out') or 'path' in name:
                    btn = QPushButton("...")
                    btn.setFixedWidth(30)
                    btn.clicked.connect(lambda _, l=line: self._pick_file(l))
                    l.addWidget(btn)

                container.widget = line

            container.setToolTip(help_text)
            self.layout.addWidget(container)
            self.controls[name] = {'widget': container.widget, 'type': ptype}

        # Spacer at bottom to push controls up
        self.layout.addStretch()
        self.updating = False

    def _pick_file(self, line_edit):
        f, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Data (*.npz *.json);;All (*.*)")
        if f:
            line_edit.setText(f)

    def get_values(self) -> Dict[str, Any]:
        out = {}
        for name, info in self.controls.items():
            w = info['widget']
            ptype = info['type']
            val = None

            if isinstance(w, QComboBox):
                val = w.currentText()
            elif isinstance(w, QCheckBox):
                val = w.isChecked()
            elif isinstance(w, QLineEdit):
                val = w.text()
            elif hasattr(w, 'value'):  # FloatParam, IntParam
                val = w.value()

            # Safety cast
            try:
                if ptype is int:
                    val = int(val)
                elif ptype is float:
                    val = float(val)
                elif ptype is bool:
                    val = bool(val)
            except:
                pass

            out[name] = val
        return out

    def _on_change(self):
        if not self.updating:
            self.paramChanged.emit()


# -----------------------------------------------------------------------------
# Main GUI Logic
# -----------------------------------------------------------------------------
class ImageGenGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procedural Image Pipeline")
        self.resize(1600, 900)

        self.original_image: Optional[Image.Image] = None
        self.processed_image: Optional[Image.Image] = None
        self.is_processing = False

        # Debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200)  # 200ms debounce
        self.update_timer.timeout.connect(self.run_pipeline)

        self._init_ui()

    def _init_ui(self):
        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        act_load = QAction("📂 Open", self)
        act_load.triggered.connect(self.load_image)
        toolbar.addAction(act_load)

        act_save = QAction("💾 Save", self)
        act_save.triggered.connect(self.save_image)
        toolbar.addAction(act_save)

        toolbar.addSeparator()
        self.chk_auto = QCheckBox("Auto-Update")
        self.chk_auto.setChecked(True)
        toolbar.addWidget(self.chk_auto)

        btn_run = QPushButton("Run Pipeline")
        btn_run.clicked.connect(self.run_pipeline)
        toolbar.addWidget(btn_run)

        # Main Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Pipeline List (Left)
        left_grp = QGroupBox("Pipeline")
        left_grp.setFixedWidth(300)
        l_layout = QVBoxLayout(left_grp)

        self.list_pipeline = QListWidget()
        self.list_pipeline.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_pipeline.model().rowsMoved.connect(self.on_pipeline_reorder)
        self.list_pipeline.currentRowChanged.connect(self.on_stage_selected)
        l_layout.addWidget(self.list_pipeline)

        add_layout = QHBoxLayout()
        self.combo_gens = QComboBox()
        self.combo_gens.addItems(REGISTRY.names())
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self.add_stage)
        add_layout.addWidget(self.combo_gens, 1)
        add_layout.addWidget(btn_add)
        l_layout.addLayout(add_layout)

        btn_rem = QPushButton("Remove Stage")
        btn_rem.clicked.connect(self.remove_stage)
        l_layout.addWidget(btn_rem)

        layout.addWidget(left_grp)

        # 2. Result View (Center)
        center_grp = QGroupBox("Result")
        c_layout = QVBoxLayout(center_grp)
        c_layout.setContentsMargins(0, 10, 0, 0)
        self.view_proc = ImageViewer()
        c_layout.addWidget(self.view_proc)
        layout.addWidget(center_grp, 1)

        # 3. Parameters (Right)
        right_grp = QGroupBox("Parameters")
        right_grp.setFixedWidth(350)
        r_layout = QVBoxLayout(right_grp)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.settings_widget = StageSettingsWidget()
        self.settings_widget.paramChanged.connect(self.on_param_changed)
        self.scroll.setWidget(self.settings_widget)
        r_layout.addWidget(self.scroll)

        layout.addWidget(right_grp)

        # Status
        self.status = QLabel("Ready")
        self.statusBar().addWidget(self.status)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            try:
                self.original_image = Image.open(path).convert("RGB")
                self.view_proc.set_image(self.original_image)
                self.processed_image = self.original_image.copy()

                # If pipeline exists, run it
                if self.list_pipeline.count() > 0:
                    self.run_pipeline()

                self.status.setText(f"Loaded {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def save_image(self):
        if not self.processed_image: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "output.png", "Images (*.png *.jpg)")
        if path:
            self.processed_image.save(path)
            self.status.setText(f"Saved to {Path(path).name}")

    def add_stage(self):
        name = self.combo_gens.currentText()
        defaults = {}
        try:
            defs = REGISTRY.create(name).get_params()
            for p in defs:
                defaults[p['name']] = p.get('default')
        except:
            pass

        data = {'name': name, 'params': defaults}
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, data)
        self.list_pipeline.addItem(item)
        self.list_pipeline.setCurrentItem(item)

        if self.chk_auto.isChecked():
            self.run_pipeline()

    def remove_stage(self):
        row = self.list_pipeline.currentRow()
        if row >= 0:
            self.list_pipeline.takeItem(row)

            # Explicitly refresh settings for remaining selection
            # Because QListWidget selection signal might not fire if row index remains same (e.g. removing index 0 of 2)
            if self.list_pipeline.count() > 0:
                current_row = self.list_pipeline.currentRow()
                self.on_stage_selected(current_row)
            else:
                self.settings_widget.clear()

            if self.chk_auto.isChecked():
                self.run_pipeline()

    def on_stage_selected(self, row):
        if row < 0:
            self.settings_widget.clear()
            return

        item = self.list_pipeline.item(row)
        data = item.data(Qt.UserRole)
        self.settings_widget.build_ui(data['name'], data['params'])

    def on_param_changed(self):
        row = self.list_pipeline.currentRow()
        if row < 0: return

        item = self.list_pipeline.item(row)
        data = item.data(Qt.UserRole)
        data['params'] = self.settings_widget.get_values()
        item.setData(Qt.UserRole, data)

        if self.chk_auto.isChecked():
            self.update_timer.start()

    def on_pipeline_reorder(self):
        if self.chk_auto.isChecked():
            self.run_pipeline()

    def run_pipeline(self):
        if self.is_processing or not self.original_image:
            return

        pipeline_specs = []
        for i in range(self.list_pipeline.count()):
            item = self.list_pipeline.item(i)
            pipeline_specs.append(item.data(Qt.UserRole))

        self.is_processing = True
        self.status.setText("Processing...")

        self.worker = ProcessingWorker(self.original_image, pipeline_specs, global_seed=None)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_finished(self, img, dt):
        self.is_processing = False
        self.processed_image = img
        self.view_proc.set_image(img)
        self.status.setText(f"Done in {dt * 1000:.1f} ms")

    def on_processing_error(self, msg):
        self.is_processing = False
        self.status.setText("Error")
        QMessageBox.warning(self, "Pipeline Error", msg)


def main():
    app = QApplication(sys.argv)
    set_dark_theme(app)
    win = ImageGenGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()