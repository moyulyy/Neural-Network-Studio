"""训练页：训练参数、后台训练线程、实时曲线与模型导出。"""

from __future__ import annotations

import math
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import torch

from ...core.persistence import save_bundle
from ...core.trainer import (
    DEVICE_LABELS,
    LOSS_LABELS,
    OPTIMIZER_LABELS,
    SCHEDULER_LABELS,
    TrainResult,
    TrainingCancelled,
    TrainingConfig,
    primary_metric_label,
    train_model,
)
from ...state import AppState
from ..theme import theme
from ..charts import ChartSeries, LineChart, ProgressRing
from ..icons import icon
from ..help import help_label
from ..widgets import (
    Card,
    ChevronComboBox,
    CardHeader,
    PageHeader,
    Pill,
    StatTile,
    Toast,
    label,
    primary_button,
    secondary_button,
)

CHART_REFRESH_MS = 90


class TrainWorker(QThread):
    epoch_ready = Signal(dict)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, model, prepared, config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.model = model
        self.prepared = prepared
        self.config = config
        self._stop = threading.Event()
        self._pause = threading.Event()

    # Control 接口
    def should_stop(self) -> bool:
        return self._stop.is_set()

    def wait_if_paused(self) -> None:
        while self._pause.is_set() and not self._stop.is_set():
            self.msleep(60)

    def pause(self) -> None:
        self._pause.set()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()

    def run(self) -> None:
        try:
            result = train_model(
                self.model,
                self.prepared,
                self.config,
                control=self,
                on_epoch=self._on_epoch,
            )
        except TrainingCancelled:
            self.cancelled.emit()
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
        else:
            self.completed.emit(result)

    def _on_epoch(self, payload: dict) -> bool:
        self.epoch_ready.emit(payload)
        return not self._stop.is_set()


class TrainPage(QWidget):
    evaluate_requested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.worker: TrainWorker | None = None
        self._result: TrainResult | None = None
        self._prepared_snapshot = None
        self._config_snapshot: TrainingConfig | None = None
        self._history: dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
            "lr": [],
        }
        self._latest: dict | None = None
        self._syncing = False
        self._chart_mode = "loss"
        self._curve_timer = QTimer(self)
        self._curve_timer.setSingleShot(True)
        self._curve_timer.setInterval(CHART_REFRESH_MS)
        self._curve_timer.timeout.connect(self._refresh_chart)
        self._build_ui()
        self.state.prepared_changed.connect(self._sync_summary)
        self.state.data_changed.connect(self._sync_summary)
        self.state.model_config_changed.connect(self._sync_summary)
        self.state.active_model_changed.connect(self._on_active_changed)
        self._sync_summary()
        self._reset_monitor()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)

        header = PageHeader("训练", "设置优化参数，实时观察损失与指标曲线")
        self.device_pill = Pill(_device_text(), "green" if torch.cuda.is_available() else "gray")
        self.export_btn = secondary_button("导出模型", "save", 36)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_model)
        header.add_action(self.device_pill)
        header.add_action(self.export_btn)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        columns = QHBoxLayout(content)
        columns.setContentsMargins(0, 0, 6, 4)
        columns.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(16)
        right = QVBoxLayout()
        right.setSpacing(14)
        columns.addLayout(left, 4)
        columns.addLayout(right, 6)

        # ---------------- 左侧：参数 ----------------
        config_card = Card(padding=14, spacing=6)
        config_card.add(CardHeader("训练参数"))
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(3)
        form.setColumnMinimumWidth(0, 100)

        self.epochs_spin = self._add_spin(
            form, 0, "训练轮数", 1, 1_000_000, 100, 10, "epochs"
        )
        self.batch_spin = self._add_spin(
            form, 1, "批大小", 1, 65536, 32, 8, "batch_size"
        )
        self.lr_edit = QLineEdit("0.001")
        self.lr_edit.setMinimumHeight(32)
        self.lr_edit.editingFinished.connect(self._normalize_lr)
        self.lr_edit.setToolTip("支持小数或科学计数法，例如 0.001 / 1e-4")
        self._add_row(form, 2, "学习率", self.lr_edit, "learning_rate")

        self.optimizer_combo = ChevronComboBox()
        for key, text in OPTIMIZER_LABELS.items():
            self.optimizer_combo.addItem(text, key)
        self._add_row(form, 3, "优化器", self.optimizer_combo, "optimizer")

        self.loss_combo = ChevronComboBox()
        for key, text in LOSS_LABELS.items():
            self.loss_combo.addItem(text, key)
        self._add_row(form, 4, "损失函数", self.loss_combo, "loss")

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 0.5)
        self.val_spin.setDecimals(2)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        self._add_row(form, 5, "验证比例", self.val_spin, "val_ratio")

        self.class_weight_combo = ChevronComboBox()
        self.class_weight_combo.addItem("无（普通交叉熵）", "none")
        self.class_weight_combo.addItem("平衡（按类别频率）", "balanced")
        self._add_row(form, 6, "类别权重", self.class_weight_combo, "class_weight")

        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.5)
        self.label_smoothing_spin.setDecimals(2)
        self.label_smoothing_spin.setSingleStep(0.05)
        self.label_smoothing_spin.setValue(0.0)
        self._add_row(
            form, 7, "标签平滑", self.label_smoothing_spin, "label_smoothing"
        )

        self.scheduler_combo = ChevronComboBox()
        for key, text in SCHEDULER_LABELS.items():
            self.scheduler_combo.addItem(text, key)
        self._add_row(form, 8, "学习率策略", self.scheduler_combo, "scheduler")

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 10.0)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setSingleStep(0.0001)
        self._add_row(form, 9, "权重衰减", self.weight_decay_spin, "weight_decay")

        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.0, 1000.0)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setSingleStep(0.5)
        self.grad_clip_spin.setToolTip("0 表示不裁剪")
        self._add_row(form, 10, "梯度裁剪", self.grad_clip_spin, "grad_clip")

        self.early_spin = self._add_spin(
            form, 11, "早停耐心", 0, 100_000, 0, 5, "early_stopping"
        )
        self.early_spin.setToolTip("验证损失连续多少轮未改善后停止，0 表示关闭")

        self.seed_spin = self._add_spin(
            form, 12, "随机种子", 0, 2_147_483_647, 42, 1, "seed"
        )

        self.device_combo = ChevronComboBox()
        for key, text in DEVICE_LABELS.items():
            self.device_combo.addItem(text, key)
        if not torch.cuda.is_available():
            index = self.device_combo.findData("cuda")
            if index >= 0:
                self.device_combo.model().item(index).setEnabled(False)
        self._add_row(form, 13, "计算设备", self.device_combo, "device")
        config_card.add_layout(form)
        left.addWidget(config_card)

        summary_card = Card(padding=12, spacing=4)
        summary_card.add(label("本次训练", "section"))
        self.summary_grid = QGridLayout()
        self.summary_grid.setHorizontalSpacing(8)
        self.summary_grid.setVerticalSpacing(5)
        self.summary_labels: dict[str, QLabel] = {}
        rows = [
            ("data", "数据"),
            ("arch", "结构"),
            ("split", "划分"),
            ("label", "标签"),
        ]
        for index, (key, title) in enumerate(rows):
            row = index // 2
            column = (index % 2) * 2
            self.summary_grid.addWidget(label(title, "tiny"), row, column)
            value = label("—", "muted")
            value.setWordWrap(True)
            self.summary_grid.addWidget(value, row, column + 1)
            self.summary_grid.setColumnStretch(column + 1, 1)
            self.summary_labels[key] = value
        summary_card.add_layout(self.summary_grid)
        left.addWidget(summary_card)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start_btn = primary_button("开始训练", "play", 40)
        self.start_btn.clicked.connect(self._start_training)
        self.pause_btn = secondary_button("暂停", "pause", 38)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn = secondary_button("停止", "stop", 38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setProperty("variant", "danger")
        self.stop_btn.clicked.connect(self._stop_training)
        buttons.addWidget(self.start_btn, 1)
        buttons.addWidget(self.pause_btn)
        buttons.addWidget(self.stop_btn)
        left.addLayout(buttons)
        left.addStretch(1)

        # ---------------- 右侧：监控 ----------------
        status_card = Card(padding=18, spacing=12)
        status_row = QHBoxLayout()
        status_row.setSpacing(18)
        self.ring = ProgressRing()
        status_row.addWidget(self.ring, 0, Qt.AlignmentFlag.AlignVCenter)
        status_text = QVBoxLayout()
        status_text.setSpacing(5)
        status_line = QHBoxLayout()
        status_line.setSpacing(8)
        self.status_pill = Pill("未开始", "gray")
        status_line.addWidget(self.status_pill)
        self.epoch_label = label("等待开始", "card-title")
        status_line.addWidget(self.epoch_label, 1)
        status_text.addLayout(status_line)
        self.run_info = label("配置参数后点击「开始训练」", "muted")
        self.run_info.setWordWrap(True)
        status_text.addWidget(self.run_info)
        self.arch_label = label("—", "mono")
        status_text.addWidget(self.arch_label)
        self.last_epoch_label = label("", "tiny")
        self.last_epoch_label.setWordWrap(True)
        status_text.addWidget(self.last_epoch_label)
        status_text.addStretch(1)
        status_row.addLayout(status_text, 1)
        status_card.add_layout(status_row)
        right.addWidget(status_card)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.tile_train = StatTile("训练损失", "—", "最后一个 epoch", "accent", "activity")
        self.tile_val = StatTile("验证损失", "—", "验证集", "purple", "target")
        self.tile_metric = StatTile("验证指标", "—", "—", "teal", "trending")
        self.tile_best = StatTile("最佳轮次", "—", "模型权重取自该轮", "orange", "clock")
        for tile in (self.tile_train, self.tile_val, self.tile_metric, self.tile_best):
            tiles.addWidget(tile, 1)
        right.addLayout(tiles)

        chart_card = Card(padding=18, spacing=8)
        self.chart_mode = _SegmentSwitch(["损失", "指标", "学习率"], self._set_chart_mode)
        chart_card.add(CardHeader("训练曲线", "点击图例可隐藏 / 显示曲线", self.chart_mode))
        self.chart = LineChart()
        self.chart.setMinimumHeight(240)
        chart_card.add(self.chart, 1)
        right.addWidget(chart_card, 1)

        finish_row = QHBoxLayout()
        finish_row.setSpacing(8)
        self.finish_hint = label("训练完成后可导出包含结构、预处理与训练信息的模型包。", "tiny")
        self.finish_hint.setWordWrap(True)
        finish_row.addWidget(self.finish_hint, 1)
        self.evaluate_btn = secondary_button("前往评估", "arrow-right", 34)
        self.evaluate_btn.setEnabled(False)
        self.evaluate_btn.clicked.connect(self.evaluate_requested.emit)
        finish_row.addWidget(self.evaluate_btn)
        right.addLayout(finish_row)

    def _add_spin(
        self,
        grid: QGridLayout,
        row: int,
        text: str,
        minimum: int,
        maximum: int,
        value: int,
        step: int,
        help_topic: str = "",
    ) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMinimumHeight(32)
        self._add_row(grid, row, text, spin, help_topic)
        return spin

    @staticmethod
    def _add_row(
        grid: QGridLayout,
        row: int,
        text: str,
        control: QWidget,
        help_topic: str = "",
    ) -> None:
        left = help_label(text, help_topic) if help_topic else label(text, "muted")
        grid.addWidget(left, row, 0)
        grid.addWidget(control, row, 1)

    def refresh_theme(self) -> None:
        self.export_btn.setIcon(icon("save", theme.palette.secondary, 16))
        self.start_btn.setIcon(icon("play", "#FFFFFF", 16))
        self.stop_btn.setIcon(icon("stop", theme.palette.red, 15))
        self.pause_btn.setIcon(
            icon("play" if self.pause_btn.text() == "继续" else "pause", theme.palette.secondary, 15)
        )
        self.evaluate_btn.setIcon(icon("arrow-right", theme.palette.secondary, 15))
        self.chart.update()

    # -- 状态同步 -----------------------------------------------------------
    def _sync_summary(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        prepared = self.state.prepared
        if prepared is None:
            self.summary_labels["data"].setText("请先准备数据")
            self.summary_labels["arch"].setText("—")
            self.summary_labels["split"].setText("—")
            self.summary_labels["label"].setText("—")
            self.start_btn.setEnabled(False)
            self.arch_label.setText("—")
            return
        self.start_btn.setEnabled(True)
        total = prepared.n_samples
        val_count = int(round(total * self.val_spin.value()))
        self.summary_labels["data"].setText(f"{total:,} 行 · {prepared.input_dim} 维输入")
        self.summary_labels["arch"].setText(
            " → ".join(
                str(x)
                for x in [
                    prepared.input_dim,
                    *self.state.model_config.hidden_layers,
                    prepared.output_dim,
                ]
            )
        )
        self.summary_labels["split"].setText(f"{total - val_count:,} 训练 / {val_count:,} 验证")
        classes = prepared.classes
        self.summary_labels["label"].setText(
            f"{prepared.preprocessor.label_name}"
            + (f" · {len(classes)} 类" if classes else " · 回归")
        )
        self.arch_label.setText(
            "网络：" + " → ".join(
                str(x)
                for x in [
                    prepared.input_dim,
                    *self.state.model_config.hidden_layers,
                    prepared.output_dim,
                ]
            )
        )
        self._sync_loss_options()

    def _sync_loss_options(self) -> None:
        task = self.state.task
        current = self.loss_combo.currentData()
        self.loss_combo.blockSignals(True)
        self.loss_combo.clear()
        if task == "classification":
            self.loss_combo.addItem("交叉熵 (CrossEntropy)", "cross_entropy")
            self.loss_combo.setEnabled(False)
        else:
            for key, text in LOSS_LABELS.items():
                self.loss_combo.addItem(text, key)
            index = self.loss_combo.findData(current)
            self.loss_combo.setCurrentIndex(max(index, 0))
            self.loss_combo.setEnabled(True)
        self.loss_combo.blockSignals(False)
        is_classification = task == "classification"
        self.class_weight_combo.setEnabled(is_classification)
        self.label_smoothing_spin.setEnabled(is_classification)

    def _normalize_lr(self) -> None:
        try:
            value = float(self.lr_edit.text().strip())
            if value <= 0:
                raise ValueError
            self.lr_edit.setText(f"{value:g}")
        except ValueError:
            self.lr_edit.setText("0.001")

    def _read_config(self) -> TrainingConfig:
        try:
            lr = float(self.lr_edit.text().strip())
        except ValueError as exc:
            raise ValueError("学习率格式无效") from exc
        loss = "cross_entropy" if self.state.task == "classification" else (
            self.loss_combo.currentData() or "auto"
        )
        is_classification = self.state.task == "classification"
        config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=lr,
            optimizer=self.optimizer_combo.currentData() or "adam",
            weight_decay=self.weight_decay_spin.value(),
            loss=loss,
            val_ratio=self.val_spin.value(),
            seed=self.seed_spin.value(),
            device=self.device_combo.currentData() or "auto",
            early_stopping=self.early_spin.value(),
            scheduler=self.scheduler_combo.currentData() or "none",
            grad_clip=self.grad_clip_spin.value(),
            class_weight=(
                self.class_weight_combo.currentData() or "none"
                if is_classification
                else "none"
            ),
            label_smoothing=(
                self.label_smoothing_spin.value() if is_classification else 0.0
            ),
        )
        config.validate()
        return config

    # -- 训练控制 -----------------------------------------------------------
    def _start_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.state.prepared is None:
            try:
                self.state.prepare_current()
            except Exception as exc:  # noqa: BLE001
                Toast.show_message(self, str(exc), "error")
                return
        try:
            config = self._read_config()
            prepared = self.state.prepare_current()
            # 在 GUI 主线程初始化随机种子，避免子线程中触发 XPU 延迟初始化
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            model = self.state.create_model()
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, str(exc), "error")
            return

        self.state.training_config = config
        self._prepared_snapshot = prepared
        self._config_snapshot = config
        self._result = None
        self._history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
            "lr": [],
        }
        self._latest = None
        self._reset_monitor()
        self._set_running_ui(True)
        self.status_pill.setText("训练中")
        self.status_pill.set_tone("accent")
        self.epoch_label.setText("正在初始化…")
        self.run_info.setText(f"{prepared.n_samples:,} 行 · {config.epochs} epochs · batch {config.batch_size}")
        self.ring.set_progress(0.0, "0%", f"Epoch 0 / {config.epochs}")
        self.last_epoch_label.setText("")
        self.finish_hint.setText("训练进行中，可随时暂停或停止；停止时会保留表现最佳的权重。")

        self.worker = TrainWorker(model, prepared, config, self)
        self.worker.epoch_ready.connect(self._on_epoch)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def _set_running_ui(self, running: bool) -> None:
        for widget in (
            self.epochs_spin,
            self.batch_spin,
            self.lr_edit,
            self.optimizer_combo,
            self.loss_combo,
            self.val_spin,
            self.scheduler_combo,
            self.weight_decay_spin,
            self.grad_clip_spin,
            self.early_spin,
            self.seed_spin,
            self.device_combo,
            self.class_weight_combo,
            self.label_smoothing_spin,
        ):
            widget.setEnabled(not running)
        if not running:
            self._sync_loss_options()
        self.start_btn.setEnabled(not running and self.state.prepared is not None)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        if running:
            self.pause_btn.setText("暂停")
            self.pause_btn.setIcon(icon("pause", theme.palette.secondary, 15))
        else:
            self.pause_btn.setText("暂停")
            self.pause_btn.setIcon(icon("pause", theme.palette.secondary, 15))

    def _toggle_pause(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        if self.worker.is_paused():
            self.worker.resume()
            self.pause_btn.setText("暂停")
            self.pause_btn.setIcon(icon("pause", theme.palette.secondary, 15))
            self.status_pill.setText("训练中")
            self.status_pill.set_tone("accent")
        else:
            self.worker.pause()
            self.pause_btn.setText("继续")
            self.pause_btn.setIcon(icon("play", theme.palette.secondary, 15))
            self.status_pill.setText("已暂停")
            self.status_pill.set_tone("orange")

    def _stop_training(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.status_pill.setText("正在停止…")
        self.status_pill.set_tone("orange")
        self.epoch_label.setText("等待当前批次结束")

    def _on_epoch(self, payload: dict) -> None:
        self._latest = payload
        for key in ("epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr"):
            if key in payload:
                self._history[key].append(float(payload[key]))
        epoch = int(payload.get("epoch", 0))
        total = int(payload.get("total_epochs", 1)) or 1
        percent = epoch / total
        self.ring.set_progress(percent, f"{percent * 100:.0f}%", f"Epoch {epoch} / {total}")
        self.epoch_label.setText(payload.get("phase_hint") or f"Epoch {epoch} / {total}")
        loss = payload.get("train_loss")
        val_loss = payload.get("val_loss")
        self.tile_train.set_value(_fmt_number(loss), "训练集")
        self.tile_val.set_value(_fmt_number(val_loss), "验证集")
        self.tile_metric.set_title("验证" + primary_metric_label(self.state.task))
        self.tile_metric.set_value(_fmt_number(payload.get("val_metric")), primary_metric_label(self.state.task))
        self.tile_best.set_value(str(payload.get("best_epoch", "—")), "最佳验证损失")
        elapsed = float(payload.get("elapsed", 0.0))
        speed = elapsed / max(epoch, 1)
        remaining = speed * max(total - epoch, 0)
        self.run_info.setText(
            f"已用时 {_fmt_time(elapsed)} · 预计剩余 {_fmt_time(remaining)} · lr {payload.get('lr', 0):.2g}"
        )
        self.last_epoch_label.setText(
            f"Epoch {epoch} · train {_fmt_number(loss)} · val {_fmt_number(val_loss)}"
            f" · best {payload.get('best_epoch', '—')}"
        )
        if not self._curve_timer.isActive():
            self._curve_timer.start()

    def _on_completed(self, result: TrainResult) -> None:
        self._result = result
        self._history = {key: list(values) for key, values in result.history.items()}
        self._latest = None
        self._set_running_ui(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        stopped = bool(result.stopped_by_user)
        self.status_pill.setText("已停止" if stopped else "训练完成")
        self.status_pill.set_tone("orange" if stopped else "green")
        self.epoch_label.setText(
            f"最佳轮次 {result.best_epoch} / {len(result.history['epoch'])}"
        )
        self.ring.set_progress(1.0, "100%", f"{len(result.history['epoch'])} epochs")
        self.run_info.setText(
            f"耗时 {_fmt_time(result.duration)} · {result.device} · 最佳轮次 {result.best_epoch}"
        )
        if result.history["train_loss"]:
            self.tile_train.set_value(_fmt_number(result.history["train_loss"][-1]), "最后一个 epoch")
        if result.best_val_loss is not None and math.isfinite(result.best_val_loss):
            self.tile_val.set_value(_fmt_number(result.best_val_loss), "最佳验证损失")
        metric_value = (
            result.final_metrics_val.get(result.metric_name)
            if result.final_metrics_val
            else result.final_metrics.get(result.metric_name)
        )
        self.tile_metric.set_title("验证" + primary_metric_label(self.state.task))
        self.tile_metric.set_value(_fmt_number(metric_value), "评估集 " + primary_metric_label(self.state.task))
        self.tile_best.set_value(str(result.best_epoch), "模型权重取自该轮")
        self._refresh_chart()
        if self._prepared_snapshot is not None and self._config_snapshot is not None:
            try:
                self.state.adopt_training_result(
                    result,
                    self._prepared_snapshot,
                    self._config_snapshot,
                    self.state.source_name(),
                )
                self.export_btn.setEnabled(True)
                self.evaluate_btn.setEnabled(True)
            except Exception as exc:  # noqa: BLE001
                Toast.show_message(self, f"保存训练结果失败：{exc}", "error")
        self.finish_hint.setText(
            "已停止训练并恢复最佳权重。" if stopped else "训练完成，模型已就绪，可导出或继续评估。"
        )
        Toast.show_message(
            self,
            "训练已停止，已保留最佳权重" if stopped else "训练完成",
            "warning" if stopped else "success",
        )
        self._cleanup_worker()

    def _on_failed(self, message: str) -> None:
        print(message, file=sys.stderr)
        self._set_running_ui(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_pill.setText("训练失败")
        self.status_pill.set_tone("red")
        self.epoch_label.setText("发生错误")
        first_line = message.strip().splitlines()[-1] if message.strip() else "未知错误"
        self.last_epoch_label.setText(first_line)
        self.finish_hint.setText("训练失败，请检查参数或数据后重试。")
        Toast.show_message(self, first_line, "error")
        self._cleanup_worker()

    def _on_cancelled(self) -> None:
        self._set_running_ui(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_pill.setText("已取消")
        self.status_pill.set_tone("gray")
        self.epoch_label.setText("训练未开始")
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.wait(2000)
            worker.deleteLater()

    def shutdown(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)

    # -- 图表 ---------------------------------------------------------------
    def _set_chart_mode(self, index: int) -> None:
        self._chart_mode = ("loss", "metric", "lr")[max(0, min(index, 2))]
        self._refresh_chart()

    def _refresh_chart(self) -> None:
        task = self.state.task
        if self._chart_mode == "loss":
            series = [
                ChartSeries("训练损失", self._history["train_loss"], "accent", fill=True),
                ChartSeries("验证损失", self._history["val_loss"], "purple", fill=False),
            ]
            self.chart.set_empty_text("开始训练后，损失曲线会显示在这里")
        elif self._chart_mode == "metric":
            label_text = primary_metric_label(task)
            series = [
                ChartSeries("训练" + label_text, self._history["train_metric"], "teal", fill=True),
                ChartSeries("验证" + label_text, self._history["val_metric"], "green", fill=False),
            ]
            self.chart.set_empty_text("开始训练后，指标曲线会显示在这里")
        else:
            series = [
                ChartSeries("学习率", self._history.get("lr", []), "orange", fill=True)
            ]
            self.chart.set_empty_text("开始训练后，学习率曲线会显示在这里")
        series = [s for s in series if any(math.isfinite(v) for v in s.values)]
        self.chart.set_series(series, self._history["epoch"])

    def _reset_monitor(self) -> None:
        self.chart.clear()
        self.tile_train.set_value("—", "最后一个 epoch")
        self.tile_val.set_value("—", "验证集")
        self.tile_metric.set_title("验证" + primary_metric_label(self.state.task))
        self.tile_metric.set_value("—", primary_metric_label(self.state.task))
        self.tile_best.set_value("—", "模型权重取自该轮")
        self.ring.set_progress(0.0, "0%", "等待开始")

    def _on_active_changed(self) -> None:
        self.export_btn.setEnabled(self.state.active is not None)
        self.evaluate_btn.setEnabled(self.state.active is not None)

    # -- 导出 ---------------------------------------------------------------
    def _export_model(self) -> None:
        active = self.state.active
        if active is None:
            Toast.show_message(self, "当前没有可导出的模型", "warning")
            return
        stem = Path(self.state.source_name()).stem or "model"
        if stem.lower() in {"iris", "housing", "churn"}:
            stem = f"neurostudio_{stem}"
        default_name = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        start_dir = self.state.source_path and str(Path(self.state.source_path).parent) or str(Path.cwd())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出模型包",
            str(Path(start_dir) / default_name),
            "NeuroStudio ZIP 模型包 (*.zip)",
        )
        if not path:
            return
        try:
            saved = save_bundle(active.bundle, path)
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"导出失败：{exc}", "error")
            return
        Toast.show_message(self, f"已导出 ZIP 模型包：{saved.name}", "success")
        self.finish_hint.setText(
            f"模型包已保存到 {saved}（ZIP 内含 weights.pt 与 model.json，包含结构、预处理、权重与训练历史）。"
        )

    # -- 展示 ---------------------------------------------------------------
    def on_show(self) -> None:
        self._sync_summary()


class _SegmentSwitch(QWidget):
    """紧凑的分段切换，用于曲线视图。"""

    def __init__(self, items: list[str], on_change, parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QButtonGroup, QPushButton

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._buttons: list[QPushButton] = []
        for index, text in enumerate(items):
            button = QPushButton(text)
            button.setProperty("variant", "chip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setChecked(index == 0)
            self._group.addButton(button, index)
            layout.addWidget(button)
            self._buttons.append(button)
        self._on_change = on_change
        self._group.idClicked.connect(self._on_change)


def _device_text() -> str:
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            return f"CUDA · {name}"
        except Exception:  # noqa: BLE001
            return "CUDA 可用"
    return "CPU"


def _fmt_number(value) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 0.01:
        return f"{number:.4f}"
    return f"{number:.5f}"


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"
