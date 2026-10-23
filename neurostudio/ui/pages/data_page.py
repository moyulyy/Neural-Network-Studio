"""数据导入与特征 / 标签配置页。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.data import (
    CATEGORICAL_LABELS,
    MISSING_LABELS,
    TASK_LABELS,
    DataOptions,
    detect_column_kind,
    infer_task,
    read_table,
)
from ...state import AppState
from ..theme import theme
from ..icons import icon, icon_pixmap
from ..help import help_label
from ..widgets import (
    Card,
    ChevronComboBox,
    CardHeader,
    DataTable,
    DropZone,
    PageHeader,
    Pill,
    Switch,
    SegmentedControl,
    Toast,
    label,
    primary_button,
    quiet_button,
    secondary_button,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = PROJECT_ROOT / "samples"


class FeatureCheckDelegate(QStyledItemDelegate):
    """自绘 iOS 风格勾选框，避免表格默认方块的视觉割裂。"""

    def paint(self, painter, option, index) -> None:
        if index.column() != 0:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        role = index.data(Qt.ItemDataRole.UserRole + 1)
        if role == "label":
            painter.setPen(QColor(theme.palette.accent))
            font = QFont(painter.font())
            font.setPixelSize(9)
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, "标签")
        else:
            state = index.data(Qt.ItemDataRole.CheckStateRole)
            checked = state in (Qt.CheckState.Checked, Qt.CheckState.Checked.value, 2)
            rect = option.rect
            box = QRectF(rect.center().x() - 9, rect.center().y() - 9, 18, 18)
            if checked:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(theme.palette.accent))
                painter.drawRoundedRect(box, 5, 5)
                pen = QPen(QColor("#FFFFFF"), 1.8)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(box.left() + 4.6, box.top() + 9.2),
                    QPointF(box.left() + 7.6, box.top() + 12.2),
                )
                painter.drawLine(
                    QPointF(box.left() + 7.6, box.top() + 12.2),
                    QPointF(box.left() + 13.4, box.top() + 5.8),
                )
            else:
                painter.setPen(QPen(QColor(theme.palette.tertiary), 1.4))
                painter.setBrush(QColor(theme.palette.card))
                painter.drawRoundedRect(box, 5, 5)
        painter.restore()


class DataPage(QWidget):
    continue_requested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._updating = False
        self._last_dir = str(Path.home())
        self._prepare_timer = QTimer(self)
        self._prepare_timer.setSingleShot(True)
        self._prepare_timer.setInterval(280)
        self._prepare_timer.timeout.connect(self._auto_prepare)
        self._hint_tone = "info"
        self._build_ui()
        self._set_enabled(False)
        self._refresh_summary()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)

        header = PageHeader("数据", "导入表格文件，选择用于训练的特征列与标签列")
        self.import_btn = primary_button("导入数据集", "upload", 36)
        self.import_btn.clicked.connect(self._choose_file)
        self.sample_btn = secondary_button("加载示例", "sparkles", 36)
        self.sample_btn.clicked.connect(self._show_samples_menu)
        header.add_action(self.sample_btn)
        header.add_action(self.import_btn)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        columns = QHBoxLayout(content)
        columns.setContentsMargins(0, 0, 6, 6)
        columns.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(16)
        right = QVBoxLayout()
        right.setSpacing(16)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)

        # ---------------- 左列：数据源 + 预览 ----------------
        source_card = Card(padding=18, spacing=12)
        self.drop_zone = DropZone("拖入 CSV / Excel 文件", "或点击选择文件", "upload")
        self.drop_zone.clicked.connect(self._choose_file)
        self.drop_zone.fileDropped.connect(self.load_file)
        source_card.add(self.drop_zone)

        self.file_row = QWidget()
        file_row = QHBoxLayout(self.file_row)
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(10)
        self.file_icon = QLabel()
        self.file_icon.setFixedSize(30, 30)
        file_row.addWidget(self.file_icon)
        file_text = QVBoxLayout()
        file_text.setSpacing(1)
        self.file_name = label("尚未导入数据", "card-title")
        self.file_path = label("支持 CSV / TSV / TXT / XLSX", "tiny")
        self.file_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_text.addWidget(self.file_name)
        file_text.addWidget(self.file_path)
        file_row.addLayout(file_text, 1)
        self.replace_btn = quiet_button("更换文件")
        self.replace_btn.clicked.connect(self._choose_file)
        file_row.addWidget(self.replace_btn)
        source_card.add(self.file_row)
        self.source_card = source_card
        left.addWidget(source_card)

        preview_card = Card(padding=18, spacing=10)
        self.preview_pill = Pill("暂无数据", "gray")
        preview_card.add(CardHeader("数据预览", "最多显示前 100 行", self.preview_pill))
        self.preview_table = DataTable()
        self.preview_table.setMinimumHeight(270)
        self.preview_table.load_dataframe(
            pd.DataFrame({"提示": ["导入数据后在此预览"]})
        )
        preview_card.add(self.preview_table)
        left.addWidget(preview_card, 1)

        # ---------------- 右列：特征与标签 ----------------
        self.config_card = Card(padding=18, spacing=12)
        self.config_card.add(CardHeader("特征与标签", "勾选参与训练的特征，并指定预测目标"))
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(3)
        self.feature_table.setHorizontalHeaderLabels(["选择", "列名", "类型"])
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setShowGrid(False)
        self.feature_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.feature_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.feature_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.feature_table.setAlternatingRowColors(False)
        self.feature_table.verticalHeader().setDefaultSectionSize(31)
        self.feature_table.horizontalHeader().setStretchLastSection(True)
        self.feature_table.setColumnWidth(0, 52)
        self.feature_table.setColumnWidth(1, 150)
        self.feature_table.setMinimumHeight(320)
        self.feature_table.setMaximumHeight(380)
        self.feature_table.itemChanged.connect(self._on_feature_item_changed)
        self.feature_table.cellClicked.connect(self._on_feature_cell_clicked)
        self.feature_table.setItemDelegateForColumn(0, FeatureCheckDelegate(self.feature_table))
        self.feature_table.viewport().setAutoFillBackground(False)
        self.config_card.add(self.feature_table)

        tools = QHBoxLayout()
        tools.setSpacing(4)
        self.select_all_btn = quiet_button("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_features(True))
        self.clear_all_btn = quiet_button("清空")
        self.clear_all_btn.clicked.connect(lambda: self._set_all_features(False))
        tools.addWidget(self.select_all_btn)
        tools.addWidget(self.clear_all_btn)
        tools.addStretch(1)
        self.selection_label = label("已选 0 / 0", "tiny")
        tools.addWidget(self.selection_label)
        self.config_card.add_layout(tools)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setContentsMargins(0, 2, 0, 0)

        self.label_combo = ChevronComboBox()
        self.label_combo.setMinimumHeight(32)
        self.label_combo.currentIndexChanged.connect(self._on_label_changed)
        form.addWidget(label("标签列", "muted"), 0, 0)
        form.addWidget(self.label_combo, 0, 1)

        form.addWidget(help_label("任务类型", "task"), 1, 0)
        self.task_control = SegmentedControl(["回归", "分类"])
        self.task_control.changed.connect(self._on_task_changed)
        form.addWidget(self.task_control, 1, 1)

        form.addWidget(help_label("缺失值", "missing"), 2, 0)
        self.missing_combo = ChevronComboBox()
        for key, text in MISSING_LABELS.items():
            self.missing_combo.addItem(text, key)
        self.missing_combo.currentIndexChanged.connect(self._on_options_changed)
        form.addWidget(self.missing_combo, 2, 1)

        form.addWidget(help_label("类别特征", "categorical"), 3, 0)
        self.categorical_combo = ChevronComboBox()
        for key, text in CATEGORICAL_LABELS.items():
            self.categorical_combo.addItem(text, key)
        self.categorical_combo.currentIndexChanged.connect(self._on_options_changed)
        form.addWidget(self.categorical_combo, 3, 1)

        form.addWidget(help_label("标准化", "scale"), 4, 0)
        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(8)
        self.scale_switch = Switch()
        self.scale_switch.setChecked(True)
        self.scale_switch.toggled.connect(self._on_options_changed)
        scale_layout.addWidget(self.scale_switch)
        scale_layout.addWidget(label("按均值 / 方差缩放", "tiny"), 1)
        form.addWidget(scale_row, 4, 1)
        self.config_card.add_layout(form)

        self.hint_label = label("", "muted")
        self.hint_label.setWordWrap(True)
        self.config_card.add(self.hint_label)
        config_actions = QHBoxLayout()
        config_actions.setSpacing(8)
        self.continue_btn = primary_button("继续配置模型", "arrow-right", 38)
        self.continue_btn.clicked.connect(self._on_continue)
        config_actions.addStretch(1)
        config_actions.addWidget(self.continue_btn)
        self.config_card.add_layout(config_actions)
        right.addWidget(self.config_card)

        # ---------------- 摘要 ----------------
        summary_card = Card(padding=16, spacing=10)
        summary_card.add(CardHeader("配置摘要", "编码后的真实输入维度"))
        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.stat_values: dict[str, QLabel] = {}
        for key, title in (
            ("rows", "样本"),
            ("features", "特征"),
            ("dims", "输入维度"),
            ("task", "任务"),
        ):
            box = QFrame()
            box.setObjectName("InsetCard")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 8, 10, 8)
            box_layout.setSpacing(1)
            box_layout.addWidget(label(title, "tiny"))
            value = label("—", "card-title")
            box_layout.addWidget(value)
            stats.addWidget(box, 1)
            self.stat_values[key] = value
        summary_card.add_layout(stats)
        self.label_summary = label("标签：—", "muted")
        summary_card.add(self.label_summary)

        right.addWidget(summary_card)
        right.addStretch(1)

        self._refresh_sample_button_icon()

    def _refresh_sample_button_icon(self) -> None:
        self.sample_btn.setIcon(icon("sparkles", theme.palette.secondary, 16))
        self.file_icon.setPixmap(icon_pixmap("file", theme.palette.tertiary, 24))

    def refresh_theme(self) -> None:
        self._refresh_sample_button_icon()
        self._update_hint_style()

    # -- 数据导入 -----------------------------------------------------------
    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入数据集",
            self._last_dir,
            "表格数据 (*.csv *.tsv *.tab *.txt *.xlsx *.xlsm);;所有文件 (*)",
        )
        if path:
            self._last_dir = str(Path(path).parent)
            self.load_file(path)

    def _show_samples_menu(self) -> None:
        menu = QMenu(self)
        examples = [
            ("鸢尾花分类（150 × 4）", "iris.csv"),
            ("房价回归（合成数据）", "housing.csv"),
            ("客户流失分类（合成数据）", "churn.csv"),
        ]
        for text, filename in examples:
            action = menu.addAction(text)
            action.triggered.connect(lambda checked=False, f=filename: self._load_sample(f))
        menu.exec(self.sample_btn.mapToGlobal(self.sample_btn.rect().bottomLeft()))

    def _load_sample(self, filename: str) -> None:
        path = SAMPLES_DIR / filename
        if not path.exists():
            Toast.show_message(self, f"示例文件不存在：{filename}", "error")
            return
        self.load_file(str(path))

    def load_file(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            frame = read_table(path)
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"读取失败：{exc}", "error")
            self._set_hint(f"读取失败：{exc}", error=True)
            return
        finally:
            QApplication.restoreOverrideCursor()

        if frame.empty or frame.shape[1] == 0:
            Toast.show_message(self, "数据文件为空", "error")
            return
        try:
            self.state.set_dataset(frame, path)
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"数据无效：{exc}", "error")
            return

        self._last_dir = str(Path(path).parent)
        self._set_enabled(True)
        self._sync_from_state()
        self._refresh_summary()
        self._schedule_prepare()
        Toast.show_message(self, f"已导入 {len(frame):,} 行 × {frame.shape[1]} 列", "success")

    def _sync_from_state(self) -> None:
        frame = self.state.frame
        if frame is None:
            return
        self._updating = True
        try:
            self.file_name.setText(self.state.source_name())
            self.file_path.setText(str(self.state.source_path or "应用内置示例"))
            self.preview_pill.setText(f"{len(frame):,} 行 × {frame.shape[1]} 列")
            self.preview_pill.set_tone("accent")
            self.preview_table.load_dataframe(frame, max_rows=100)

            self.label_combo.clear()
            self.label_combo.addItems([str(c) for c in frame.columns])
            index = self.label_combo.findText(self.state.label_column)
            if index >= 0:
                self.label_combo.setCurrentIndex(index)

            self.task_control.set_current_index(0 if self.state.task == "regression" else 1)
            self._set_combo_data(self.missing_combo, self.state.options.missing)
            self._set_combo_data(self.categorical_combo, self.state.options.categorical)
            self.scale_switch.setChecked(self.state.options.scale)
            self._rebuild_feature_table()
        finally:
            self._updating = False
        self._update_counts()
        self._update_estimates()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _rebuild_feature_table(self) -> None:
        frame = self.state.frame
        if frame is None:
            return
        was_updating = self._updating
        self._updating = True
        try:
            self.feature_table.setRowCount(len(frame.columns))
            for row, column in enumerate(frame.columns):
                name = str(column)
                selected = name in self.state.feature_columns
                is_label = name == self.state.label_column
                check_item = QTableWidgetItem()
                check_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if not is_label:
                    check_item.setData(
                        Qt.ItemDataRole.CheckStateRole,
                        Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked,
                    )
                check_item.setData(Qt.ItemDataRole.UserRole, name)
                check_item.setData(
                    Qt.ItemDataRole.UserRole + 1, "label" if is_label else "feature"
                )
                check_item.setToolTip(
                    "标签列不可作为特征" if is_label else "点击切换是否作为特征"
                )
                self.feature_table.setItem(row, 0, check_item)

                name_item = QTableWidgetItem(name)
                if is_label:
                    font = QFont(name_item.font())
                    font.setBold(True)
                    name_item.setFont(font)
                self.feature_table.setItem(row, 1, name_item)

                kind = detect_column_kind(frame[column])
                kind_text = "数值" if kind == "numeric" else "类别"
                kind_item = QTableWidgetItem(kind_text)
                kind_item.setForeground(QColor(theme.palette.secondary))
                self.feature_table.setItem(row, 2, kind_item)
                if is_label:
                    for col in range(3):
                        item = self.feature_table.item(row, col)
                        if item is not None:
                            item.setForeground(QColor(theme.palette.accent))
            self.feature_table.resizeColumnsToContents()
            self.feature_table.setColumnWidth(0, 56)
        finally:
            self._updating = was_updating

    def _on_feature_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 0:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return
        if name == self.state.label_column:
            return
        state = item.data(Qt.ItemDataRole.CheckStateRole)
        checked = state in (Qt.CheckState.Checked, Qt.CheckState.Checked.value, 2)
        if checked:
            if name not in self.state.feature_columns:
                self.state.feature_columns.append(name)
        else:
            if name in self.state.feature_columns:
                self.state.feature_columns.remove(name)
        self._selection_changed()

    def _on_feature_cell_clicked(self, row: int, column: int) -> None:
        if self._updating:
            return
        item = self.feature_table.item(row, 0)
        if item is None:
            return
        if item.data(Qt.ItemDataRole.UserRole + 1) == "label":
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name or name == self.state.label_column:
            return
        state = item.data(Qt.ItemDataRole.CheckStateRole)
        checked = state in (Qt.CheckState.Checked, Qt.CheckState.Checked.value, 2)
        item.setData(
            Qt.ItemDataRole.CheckStateRole,
            Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked,
        )

    def _set_all_features(self, checked: bool) -> None:
        if self.state.frame is None:
            return
        self._updating = True
        try:
            for row in range(self.feature_table.rowCount()):
                item = self.feature_table.item(row, 0)
                name = item.data(Qt.ItemDataRole.UserRole) if item else None
                if not name or name == self.state.label_column:
                    continue
                item.setData(
                    Qt.ItemDataRole.CheckStateRole,
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                )
        finally:
            self._updating = False
        if checked:
            self.state.feature_columns = [
                str(c) for c in self.state.frame.columns if str(c) != self.state.label_column
            ]
        else:
            self.state.feature_columns = []
        self._selection_changed()

    def _selection_changed(self) -> None:
        self._update_counts()
        self._update_estimates()
        self.state.invalidate()
        self.state.data_changed.emit()
        self._schedule_prepare()

    def _on_label_changed(self, index: int) -> None:
        if self._updating or self.state.frame is None or index < 0:
            return
        label_name = self.label_combo.itemText(index)
        if label_name == self.state.label_column:
            return
        self.state.label_column = label_name
        if label_name in self.state.feature_columns:
            self.state.feature_columns.remove(label_name)
        inferred = infer_task(self.state.frame[label_name])
        self.state.task = inferred
        self._updating = True
        try:
            self.task_control.set_current_index(0 if inferred == "regression" else 1)
            self._rebuild_feature_table()
        finally:
            self._updating = False
        self._update_counts()
        self._update_estimates()
        self.state.invalidate()
        self.state.data_changed.emit()
        self._schedule_prepare()

    def _on_task_changed(self, index: int) -> None:
        if self._updating:
            return
        self.state.task = "regression" if index == 0 else "classification"
        self.state.invalidate()
        self.state.data_changed.emit()
        self._update_estimates()
        self._schedule_prepare()

    def _on_options_changed(self) -> None:
        if self._updating:
            return
        self.state.options = DataOptions(
            missing=self.missing_combo.currentData() or "median",
            scale=self.scale_switch.isChecked(),
            categorical=self.categorical_combo.currentData() or "onehot",
        )
        self.state.invalidate()
        self.state.data_changed.emit()
        self._update_estimates()
        self._schedule_prepare()

    # -- 自动准备 -----------------------------------------------------------
    def _schedule_prepare(self) -> None:
        if self.state.frame is not None and self.state.feature_columns:
            self._prepare_timer.start()

    def _auto_prepare(self) -> None:
        if self.state.frame is None or not self.state.feature_columns:
            self._update_estimates()
            return
        try:
            prepared = self.state.prepare_current()
        except Exception as exc:  # noqa: BLE001
            self.state.prepared = None
            self.stat_values["dims"].setText("—")
            self._set_hint(str(exc), error=True)
            return
        self.stat_values["dims"].setText(str(prepared.input_dim))
        classes = prepared.classes
        label_text = f"标签：{self.state.label_column}"
        if classes:
            label_text += f" · {len(classes)} 个类别"
            preview = "、".join(classes[:5]) + ("…" if len(classes) > 5 else "")
            label_text += f"（{preview}）"
        self.label_summary.setText(label_text)
        if prepared.dropped_rows:
            self._set_hint(
                f"有 {prepared.dropped_rows} 行因缺失值或无效标签被丢弃，实际参与训练 {prepared.n_samples} 行。",
                warning=True,
            )
        else:
            self._set_hint("配置已就绪，可以继续调整网络结构。")
        self.continue_btn.setEnabled(True)
        self.state.data_changed.emit()

    def _on_continue(self) -> None:
        if self.state.frame is None:
            Toast.show_message(self, "请先导入数据集", "error")
            return
        if not self.state.feature_columns:
            Toast.show_message(self, "请至少选择一个特征列", "error")
            return
        try:
            self.state.prepare_current()
        except Exception as exc:  # noqa: BLE001
            self._set_hint(str(exc), error=True)
            Toast.show_message(self, str(exc), "error")
            return
        self.continue_btn.setEnabled(True)
        self.continue_requested.emit()

    # -- 摘要与提示 ---------------------------------------------------------
    def _update_counts(self) -> None:
        total = 0 if self.state.frame is None else len(self.state.frame.columns) - 1
        self.selection_label.setText(f"已选 {len(self.state.feature_columns)} / {total}")

    def _update_estimates(self) -> None:
        if self.state.frame is None:
            return
        self.stat_values["rows"].setText(f"{len(self.state.frame):,}")
        self.stat_values["features"].setText(str(len(self.state.feature_columns)))
        self.stat_values["dims"].setText(str(self.state.estimated_input_dim()))
        self.stat_values["task"].setText(TASK_LABELS.get(self.state.task, self.state.task))
        self.label_summary.setText(f"标签：{self.state.label_column or '—'}")

    def _refresh_summary(self) -> None:
        if self.state.frame is None:
            self.stat_values["rows"].setText("—")
            self.stat_values["features"].setText("—")
            self.stat_values["dims"].setText("—")
            self.stat_values["task"].setText("—")

    def _set_hint(self, text: str, error: bool = False, warning: bool = False) -> None:
        self.hint_label.setText(text)
        self._hint_tone = "error" if error else ("warning" if warning else "info")
        self._update_hint_style()

    def _update_hint_style(self) -> None:
        color = {
            "error": theme.palette.red,
            "warning": theme.palette.orange,
        }.get(self._hint_tone, theme.palette.tertiary)
        self.hint_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _set_enabled(self, enabled: bool) -> None:
        self.config_card.setEnabled(enabled)
        self.continue_btn.setEnabled(enabled)
        self.replace_btn.setEnabled(enabled)
        if not enabled:
            self.file_name.setText("尚未导入数据")
            self.file_path.setText("支持 CSV / TSV / TXT / XLSX")

