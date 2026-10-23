"""NeuroStudio 独立预测窗口。

只负责：加载导出的模型 ZIP（weights.pt + model.json），根据 model.json 中的
特征定义生成输入框，输入一组特征值后给出推理结果。不包含训练功能。

启动：
    python tests/app/predict_app.py
    python tests/app/predict_app.py path/to/model.zip
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from neurostudio.core.persistence import (
    bundle_summary,
    load_bundle,
    model_from_bundle,
    preprocessor_from_bundle,
)
from neurostudio.core.trainer import forward_predict
from neurostudio.ui.theme import theme
from neurostudio.ui.widgets import (
    Card,
    CardHeader,
    ChevronComboBox,
    DropZone,
    PageHeader,
    Pill,
    label,
    primary_button,
)

MODEL_FILTER = "NeuroStudio 模型包 (*.zip *.nfs *.pt);;所有文件 (*)"


class PredictWindow(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("NeuroStudio · 快速预测")
        self.resize(640, 880)
        self.setMinimumSize(540, 720)

        self.bundle: dict | None = None
        self.model = None
        self.preprocessor = None
        self.fields: list[tuple[object, QWidget]] = []
        self._last_dir = str(Path.home())

        self._build_ui()
        self._show_empty_form()
        self._set_status("请先加载模型 ZIP。")
        self.predict_btn.setEnabled(False)

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(12)

        header = PageHeader("快速预测", "加载模型包，输入一组特征值即可得到推理结果")
        header.add_action(Pill("仅推理", "purple"))
        outer.addWidget(header)

        model_card = Card(padding=14, spacing=7)
        self.model_pill = Pill("未加载", "gray")
        model_card.add(CardHeader("模型包", "ZIP 内含 weights.pt 与 model.json", self.model_pill))
        self.drop_zone = DropZone("拖入 ZIP 模型包", "或点击选择；兼容旧版 .nfs", "folder")
        self.drop_zone.setMinimumHeight(76)
        self.drop_zone.clicked.connect(self.choose_file)
        self.drop_zone.fileDropped.connect(self.load_model)
        model_card.add(self.drop_zone)

        file_row = QHBoxLayout()
        file_row.setSpacing(10)
        self.file_name = label("尚未加载模型", "card-title")
        file_row.addWidget(self.file_name, 1)
        self.file_meta = label("", "tiny")
        self.file_meta.setToolTip("")
        file_row.addWidget(self.file_meta, 0, Qt.AlignmentFlag.AlignRight)
        model_card.add_layout(file_row)

        self.model_meta = label("", "muted")
        self.model_meta.setWordWrap(True)
        model_card.add(self.model_meta)
        self.train_meta = label("", "tiny")
        self.train_meta.setWordWrap(True)
        model_card.add(self.train_meta)
        self.status_label = label("", "tiny")
        self.status_label.setWordWrap(True)
        model_card.add(self.status_label)
        outer.addWidget(model_card)

        features_card = Card(padding=16, spacing=8)
        self.feature_pill = Pill("—", "gray")
        features_card.add(
            CardHeader("输入特征", "数值特征已预填训练集均值，可直接修改", self.feature_pill)
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form_host = QWidget()
        self.form = QGridLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(8)
        self.form.setColumnMinimumWidth(0, 132)
        self.form.setColumnStretch(1, 1)
        self.form.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.feature_scroll = scroll
        scroll.setWidget(self.form_host)
        scroll.setMinimumHeight(170)
        features_card.add(scroll, 1)
        outer.addWidget(features_card, 1)

        result_card = Card(padding=16, spacing=5)
        result_card.add(CardHeader("推理结果", "模型输出"))
        self.result_value = label("—", "value")
        self.result_value.setStyleSheet(
            f"color: {theme.palette.accent}; font-size: 30px; font-weight: 700;"
        )
        result_card.add(self.result_value)
        self.result_detail = label("加载模型并输入特征后点击预测。", "muted")
        self.result_detail.setWordWrap(True)
        result_card.add(self.result_detail)
        self.result_extra = label("", "tiny")
        self.result_extra.setWordWrap(True)
        result_card.add(self.result_extra)
        outer.addWidget(result_card)

        self.predict_btn = primary_button("开始预测", "arrow-right", 44)
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setDefault(True)
        outer.addWidget(self.predict_btn)

    # -- 模型 -------------------------------------------------------------
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模型包", self._last_dir, MODEL_FILTER
        )
        if path:
            self._last_dir = str(Path(path).parent)
            self.load_model(path)

    def load_model(self, path: str) -> None:
        try:
            bundle = load_bundle(path)
            model = model_from_bundle(bundle)
            preprocessor = preprocessor_from_bundle(bundle)
            if not preprocessor.features:
                raise ValueError("模型包没有记录输入特征")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"加载失败：{exc}", error=True)
            return

        self.bundle = bundle
        self.model = model
        self.preprocessor = preprocessor
        self.drop_zone.set_text("拖入或点击更换模型包", f"已加载：{Path(path).name}")
        self.file_name.setText(Path(path).name)
        self.file_meta.setText(f"{Path(path).stat().st_size / 1024:.1f} KB")
        self.file_meta.setToolTip(str(path))

        summary = bundle_summary(bundle)
        self.model_pill.setText(summary["task_label"])
        self.model_pill.set_tone(
            "green" if summary["task"] == "classification" else "accent"
        )
        classes = summary["classes"]
        class_text = f" · {len(classes)} 个类别" if classes else ""
        self.model_meta.setText(
            f"结构 {summary['architecture']} · {summary['task_label']} · "
            f"标签「{summary['label']}」{class_text}"
        )
        metric_label = "准确率" if summary["task"] == "classification" else "R²"
        metric_value = float(summary["primary_metric"])
        metric_text = (
            f"{metric_value * 100:.2f}%"
            if summary["task"] == "classification"
            else f"{metric_value:.4f}"
        )
        self.train_meta.setText(
            f"{summary['epochs']} epochs · 最佳轮次 {summary['best_epoch']} · "
            f"训练{metric_label} {metric_text} · {summary['device']}"
        )
        self.feature_pill.setText(f"{len(preprocessor.features)} 个特征")
        self._build_form()
        self.predict_btn.setEnabled(True)
        self.result_value.setText("—")
        self.result_value.setStyleSheet(
            f"color: {theme.palette.accent}; font-size: 30px; font-weight: 700;"
        )
        self.result_detail.setText("输入特征后点击「开始预测」。")
        self.result_extra.setText("")
        self._set_status("模型已加载，可开始预测。", success=True)

    # -- 表单 -------------------------------------------------------------
    def _clear_form(self) -> None:
        while self.form.count():
            item = self.form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.fields.clear()

    def _show_empty_form(self) -> None:
        self._clear_form()
        hint = label("加载模型包后，这里会根据训练时的特征自动生成输入框。", "muted")
        hint.setWordWrap(True)
        self.form.addWidget(hint, 0, 0, 1, 2)

    def _build_form(self) -> None:
        self._clear_form()
        if self.preprocessor is None:
            return
        for row, spec in enumerate(self.preprocessor.specs):
            name_label = label(spec.name, "muted")
            name_label.setToolTip(
                "数值特征" if spec.kind == "numeric" else "类别特征"
            )
            self.form.addWidget(name_label, row, 0)
            if spec.kind == "numeric":
                edit = QLineEdit()
                value = spec.mean if math.isfinite(spec.mean) else spec.fill
                edit.setText(f"{value:.6g}")
                edit.setPlaceholderText("输入数值")
                edit.setMinimumHeight(32)
                edit.setToolTip("已预填训练集均值，可修改")
                self.form.addWidget(edit, row, 1)
                self.fields.append((spec, edit))
            else:
                combo = ChevronComboBox()
                combo.addItems(spec.categories)
                index = combo.findText(spec.fill_label)
                if index >= 0:
                    combo.setCurrentIndex(index)
                combo.setMinimumHeight(32)
                self.form.addWidget(combo, row, 1)
                self.fields.append((spec, combo))

    # -- 推理 -------------------------------------------------------------
    def predict(self) -> None:
        if self.model is None or self.preprocessor is None:
            self._set_status("请先加载模型包。", error=True)
            return
        row: dict[str, object] = {}
        for spec, widget in self.fields:
            if spec.kind == "numeric":
                text = widget.text().strip().replace(",", "")
                if not text:
                    row[spec.name] = spec.fill
                    continue
                try:
                    value = float(text)
                except ValueError:
                    self._set_status(f"「{spec.name}」请输入有效数值。", error=True)
                    return
                if not math.isfinite(value):
                    self._set_status(f"「{spec.name}」请输入有限数值。", error=True)
                    return
                row[spec.name] = value
            else:
                row[spec.name] = widget.currentText()

        try:
            frame = pd.DataFrame([row], columns=self.preprocessor.features)
            matrix = self.preprocessor.transform(frame)
            prediction, probabilities, _ = forward_predict(
                self.model, matrix, self.preprocessor.task
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"预测失败：{exc}", error=True)
            return

        if len(prediction) == 0:
            self._set_status("没有得到有效预测。", error=True)
            return

        if self.preprocessor.task == "classification":
            index = int(prediction[0])
            classes = self.preprocessor.classes
            name = classes[index] if 0 <= index < len(classes) else str(index)
            confidence = (
                float(probabilities[0, index]) * 100.0
                if probabilities is not None and 0 <= index < probabilities.shape[1]
                else 0.0
            )
            self.result_value.setText(name)
            self.result_detail.setText(
                f"置信度 {confidence:.2f}% · 输出维度 {self.preprocessor.output_dim}"
            )
            if probabilities is not None:
                pairs = sorted(
                    (
                        (classes[i], float(probabilities[0, i]) * 100.0)
                        for i in range(min(len(classes), probabilities.shape[1]))
                    ),
                    key=lambda item: -item[1],
                )
                shown = pairs[:6]
                suffix = " …" if len(pairs) > len(shown) else ""
                self.result_extra.setText(
                    "   ".join(f"{label} {p:.2f}%" for label, p in shown) + suffix
                )
            else:
                self.result_extra.setText("")
        else:
            value = float(prediction[0])
            self.result_value.setText(_fmt_number(value))
            self.result_detail.setText(f"标签：{self.preprocessor.label_name}")
            self.result_extra.setText("")
        self._set_status("预测完成。", success=True)

    # -- 状态 -------------------------------------------------------------
    def _set_status(self, text: str, error: bool = False, success: bool = False) -> None:
        self.status_label.setText(text)
        color = (
            theme.palette.red
            if error
            else (theme.palette.green if success else theme.palette.tertiary)
        )
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px;")


def _fmt_number(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if abs(value) >= 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroStudio Predict")
    app.setApplicationDisplayName("NeuroStudio 快速预测")
    app.setStyle("Fusion")
    theme.set_mode("light", notify=False)

    window = PredictWindow()
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            window.load_model(str(candidate))
        else:
            window._set_status(f"文件不存在：{candidate}", error=True)
    screen = app.primaryScreen()
    if screen is not None:
        geometry = screen.availableGeometry()
        window.move(
            geometry.left() + (geometry.width() - window.width()) // 2,
            geometry.top() + (geometry.height() - window.height()) // 2,
        )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
