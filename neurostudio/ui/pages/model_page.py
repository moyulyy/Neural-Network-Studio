"""网络结构配置页：隐藏层、激活函数、正则化与结构预览。"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.model import (
    ACTIVATION_LABELS,
    INITIALIZERS,
    MAX_HIDDEN_LAYERS,
    estimate_model_size,
)
from ...core.trainer import primary_metric_label
from ...state import AppState
from ..theme import theme
from ..charts import NetworkDiagram
from ..icons import icon
from ..help import HelpButton, help_label
from ..widgets import (
    Card,
    ChevronComboBox,
    CardHeader,
    PageHeader,
    Pill,
    Switch,
    Toast,
    icon_button,
    label,
    primary_button,
    secondary_button,
)


class ModelPage(QWidget):
    continue_requested = Signal()
    back_requested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._row_widgets: list[QWidget] = []
        self._syncing = False
        self._build_ui()
        self.state.prepared_changed.connect(self._sync_structure)
        self.state.data_changed.connect(self._sync_structure)
        self.state.model_config_changed.connect(self._sync_structure)
        self._refresh_layers()
        self._sync_structure()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        header = PageHeader("模型", "自定义输入维度、隐藏层规模与训练相关的网络参数")
        self.structure_pill = Pill("多层感知机", "purple")
        header.add_action(self.structure_pill)
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
        columns.addLayout(left, 5)
        columns.addLayout(right, 4)

        # 结构预览
        preview_card = Card(padding=18, spacing=10)
        self.diagram_pill = Pill("等待数据", "gray")
        preview_card.add(
            CardHeader("网络结构", "按当前特征编码与隐藏层设置实时绘制", self.diagram_pill)
        )
        self.diagram = NetworkDiagram()
        self.diagram.setMinimumHeight(300)
        preview_card.add(self.diagram, 1)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.arch_label = label("—", "mono")
        footer.addWidget(self.arch_label, 1)
        self.param_pill = Pill("0 参数", "gray")
        footer.addWidget(self.param_pill)
        preview_card.add_layout(footer)
        left.addWidget(preview_card, 1)

        # 隐藏层编辑
        layers_card = Card(padding=18, spacing=10)
        self.add_layer_btn = secondary_button("添加隐藏层", "plus", 32)
        self.add_layer_btn.clicked.connect(self._add_layer)
        layer_tools = QWidget()
        layer_tools_layout = QHBoxLayout(layer_tools)
        layer_tools_layout.setContentsMargins(0, 0, 0, 0)
        layer_tools_layout.setSpacing(4)
        layer_tools_layout.addWidget(HelpButton("structure"))
        layer_tools_layout.addWidget(self.add_layer_btn)
        layers_card.add(
            CardHeader(
                "隐藏层",
                f"最多 {MAX_HIDDEN_LAYERS} 层，每层神经元数可独立设置",
                layer_tools,
            )
        )
        self.layers_host = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_host)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        self.layers_layout.setSpacing(7)
        layers_card.add(self.layers_host)
        left.addWidget(layers_card)

        # 网络参数
        params_card = Card(padding=16, spacing=10)
        params_card.add(CardHeader("网络参数", "影响模型的表达能力和训练稳定性"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnMinimumWidth(0, 86)

        grid.addWidget(help_label("激活函数", "activation"), 0, 0)
        self.activation_combo = ChevronComboBox()
        for key, text in ACTIVATION_LABELS.items():
            self.activation_combo.addItem(text, key)
        self.activation_combo.currentIndexChanged.connect(self._on_activation_changed)
        grid.addWidget(self.activation_combo, 0, 1)

        grid.addWidget(help_label("权重初始化", "init"), 1, 0)
        self.init_combo = ChevronComboBox()
        for key, text in INITIALIZERS.items():
            self.init_combo.addItem(text, key)
        self.init_combo.currentIndexChanged.connect(self._on_init_changed)
        grid.addWidget(self.init_combo, 1, 1)

        grid.addWidget(help_label("批归一化", "batch_norm"), 2, 0)
        bn_row = QWidget()
        bn_layout = QHBoxLayout(bn_row)
        bn_layout.setContentsMargins(0, 0, 0, 0)
        bn_layout.setSpacing(8)
        self.bn_switch = Switch()
        self.bn_switch.toggled.connect(self._on_batch_norm_changed)
        bn_layout.addWidget(self.bn_switch)
        bn_layout.addWidget(label("每个隐藏层后插入 BatchNorm", "tiny"), 1)
        grid.addWidget(bn_row, 2, 1)

        grid.addWidget(help_label("Dropout", "dropout"), 3, 0)
        dropout_row = QWidget()
        dropout_layout = QHBoxLayout(dropout_row)
        dropout_layout.setContentsMargins(0, 0, 0, 0)
        dropout_layout.setSpacing(10)
        self.dropout_slider = QSlider(Qt.Orientation.Horizontal)
        self.dropout_slider.setRange(0, 80)
        self.dropout_slider.setSingleStep(1)
        self.dropout_slider.setPageStep(5)
        self.dropout_slider.valueChanged.connect(self._on_dropout_changed)
        dropout_layout.addWidget(self.dropout_slider, 1)
        self.dropout_value = label("0%", "muted")
        self.dropout_value.setMinimumWidth(34)
        self.dropout_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dropout_layout.addWidget(self.dropout_value)
        grid.addWidget(dropout_row, 3, 1)
        params_card.add_layout(grid)
        right.addWidget(params_card)

        # 结构信息
        info_card = Card(padding=16, spacing=8)
        info_card.add(CardHeader("结构摘要", "训练开始前确认维度与参数规模", HelpButton("structure")))
        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(12)
        info_grid.setVerticalSpacing(7)
        self.info_labels: dict[str, QLabel] = {}
        rows = [
            ("input", "输入维度"),
            ("output", "输出维度"),
            ("hidden", "隐藏层"),
            ("params", "可训练参数"),
            ("task", "任务类型"),
            ("label", "标签列"),
        ]
        for index, (key, title) in enumerate(rows):
            info_grid.addWidget(label(title, "muted"), index, 0)
            value = label("—", "card-title")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setWordWrap(True)
            info_grid.addWidget(value, index, 1)
            self.info_labels[key] = value
        info_card.add_layout(info_grid)
        self.output_hint = label("", "tiny")
        self.output_hint.setWordWrap(True)
        info_card.add(self.output_hint)
        right.addWidget(info_card)

        # 模型尺寸估算
        size_card = Card(padding=16, spacing=6)
        size_card.add(
            CardHeader("模型尺寸估算", "权重、存储与显存占用", HelpButton("model_size"))
        )
        size_grid = QGridLayout()
        size_grid.setHorizontalSpacing(12)
        size_grid.setVerticalSpacing(4)
        self.size_labels: dict[str, QLabel] = {}
        size_rows = [
            ("weights", "权重元素（不含偏置）"),
            ("params", "参数总数（含偏置）"),
            ("storage", "模型存储"),
            ("infer", "推理显存"),
            ("train", "训练显存（张量估算）"),
        ]
        for index, (key, title) in enumerate(size_rows):
            size_grid.addWidget(label(title, "muted"), index, 0)
            value = label("—", "card-title")
            value.setWordWrap(True)
            size_grid.addWidget(value, index, 1)
            self.size_labels[key] = value
        size_grid.setColumnStretch(1, 1)
        size_card.add_layout(size_grid)
        self.size_hint = label("选择数据后估算模型尺寸。", "tiny")
        self.size_hint.setWordWrap(True)
        size_card.add(self.size_hint)
        right.addWidget(size_card)
        right.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.back_btn = secondary_button("返回数据", "chevron-left", 38)
        self.back_btn.clicked.connect(self.back_requested.emit)
        self.continue_btn = primary_button("继续配置训练", "arrow-right", 38)
        self.continue_btn.clicked.connect(self._on_continue)
        actions.addWidget(self.back_btn)
        actions.addStretch(1)
        actions.addWidget(self.continue_btn)
        outer.addLayout(actions)

    # -- 同步 -------------------------------------------------------------
    def _sync_structure(self) -> None:
        config = self.state.model_config
        prepared = self.state.prepared
        has_data = self.state.frame is not None
        input_dim = prepared.input_dim if prepared else (config.input_dim if has_data else 0)
        output_dim = prepared.output_dim if prepared else (config.output_dim if has_data else 0)
        size_config = replace(
            config,
            input_dim=max(int(input_dim), 1),
            hidden_layers=list(config.hidden_layers),
            output_dim=max(int(output_dim), 1),
        )
        size_stats = estimate_model_size(size_config)

        self._syncing = True
        try:
            self.bn_switch.setChecked(config.batch_norm)
            self.dropout_slider.setValue(int(round(config.dropout * 100)))
            activation_index = self.activation_combo.findData(config.activation)
            if activation_index >= 0:
                self.activation_combo.setCurrentIndex(activation_index)
            init_index = self.init_combo.findData(config.init)
            if init_index >= 0:
                self.init_combo.setCurrentIndex(init_index)
        finally:
            self._syncing = False
        self.dropout_value.setText(f"{int(round(config.dropout * 100))}%")

        if has_data and prepared is not None:
            layers = [
                {"kind": "input", "label": "输入", "width": input_dim},
                *[
                    {
                        "kind": "hidden",
                        "label": f"隐藏 {i}",
                        "width": width,
                        "activation": ACTIVATION_LABELS.get(config.activation, config.activation),
                        "dropout": config.dropout,
                        "batch_norm": config.batch_norm,
                    }
                    for i, width in enumerate(config.hidden_layers, start=1)
                ],
                {"kind": "output", "label": "输出", "width": output_dim},
            ]
            self.diagram.set_layers(layers)
            self.diagram_pill.setText(f"{len(config.hidden_layers) + 2} 层 · MLP")
            self.diagram_pill.set_tone("purple")
            self.structure_pill.setText(f"MLP · {len(config.hidden_layers)} 隐藏层")
            self.structure_pill.set_tone("purple")
            self.arch_label.setText(
                " → ".join(str(x) for x in [input_dim, *config.hidden_layers, output_dim])
            )
            self.info_labels["input"].setText(str(input_dim))
            self.info_labels["output"].setText(str(output_dim))
            self.info_labels["hidden"].setText(
                " · ".join(str(x) for x in config.hidden_layers)
            )
            parameters = size_stats.parameter_count
            self.info_labels["params"].setText(_format_params(parameters))
            self.param_pill.setText(f"{_format_params(parameters)} 参数")
            self.param_pill.set_tone("accent")
        else:
            self.diagram.set_layers([])
            self.diagram_pill.setText("等待数据" if not has_data else "待准备")
            self.diagram_pill.set_tone("gray")
            self.structure_pill.setText("多层感知机")
            self.structure_pill.set_tone("purple")
            self.arch_label.setText("—")
            self.info_labels["input"].setText("—")
            self.info_labels["output"].setText("—")
            self.info_labels["hidden"].setText(" · ".join(str(x) for x in config.hidden_layers))
            parameters = size_stats.parameter_count
            self.info_labels["params"].setText(_format_params(parameters) if has_data else "—")
            self.param_pill.setText(f"{_format_params(parameters)} 参数" if has_data else "0 参数")
            self.param_pill.set_tone("gray")

        self.info_labels["task"].setText("回归" if self.state.task == "regression" else "分类")
        self.info_labels["label"].setText(self.state.label_column or "—")
        if self.state.task == "classification":
            classes = prepared.classes if prepared else []
            self.output_hint.setText(
                f"分类输出层自动等于标签类别数"
                + (f"（{len(classes)} 类：{'、'.join(classes[:4])}{'…' if len(classes) > 4 else ''}）" if classes else "")
                + "，以保证交叉熵标签对齐。"
            )
        else:
            self.output_hint.setText("回归任务使用单输出神经元，输出值即预测值。")
        self.continue_btn.setEnabled(has_data)
        self.add_layer_btn.setEnabled(len(config.hidden_layers) < MAX_HIDDEN_LAYERS)
        self._update_size_card(size_stats, has_data and prepared is not None)

    def _update_size_card(self, stats, ready: bool) -> None:
        if not ready:
            for value in self.size_labels.values():
                value.setText("—")
            self.size_hint.setText("选择数据后估算模型尺寸。")
            return
        cfg = self.state.training_config
        batch = max(1, int(cfg.batch_size))
        optimizer = str(cfg.optimizer).upper()
        param_bytes = stats.fp32_bytes
        activation_bytes = stats.activation_elements * 4
        optimizer_states = {"ADAM": 2, "ADAMW": 2, "SGD": 1, "RMSPROP": 1}.get(
            optimizer, 2
        )
        train_bytes = int(
            (
                param_bytes
                + param_bytes
                + param_bytes * optimizer_states
                + activation_bytes * 2 * batch
            )
            * 1.15
        )
        self.size_labels["weights"].setText(f"{stats.weight_count:,}")
        self.size_labels["params"].setText(f"{stats.parameter_count:,}")
        self.size_labels["storage"].setText(
            f"{_format_bytes(stats.fp32_bytes)} (FP32) · "
            f"{_format_bytes(stats.fp16_bytes)} (FP16)"
        )
        self.size_labels["infer"].setText(
            f"{_format_bytes(param_bytes + activation_bytes)} · batch 1"
        )
        self.size_labels["train"].setText(
            f"{_format_bytes(train_bytes)} · batch {batch} · {optimizer}"
        )
        self.size_hint.setText(
            f"按 FP32、batch {batch}、{optimizer} 估算；不含 CUDA 上下文与框架缓存。"
        )


    # -- 隐藏层行 -----------------------------------------------------------
    def _clear_layer_rows(self) -> None:
        for widget in self._row_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._row_widgets.clear()
        while self.layers_layout.count():
            item = self.layers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _refresh_layers(self) -> None:
        self._clear_layer_rows()
        config = self.state.model_config
        for index, width in enumerate(config.hidden_layers):
            row = QFrame()
            row.setObjectName("InsetCard")
            layout = QHBoxLayout(row)
            layout.setContentsMargins(10, 6, 8, 6)
            layout.setSpacing(10)
            index_label = label(f"隐藏层 {index + 1}", "muted")
            index_label.setMinimumWidth(68)
            layout.addWidget(index_label)
            spin = QSpinBox()
            spin.setRange(1, 8192)
            spin.setSingleStep(8)
            spin.setValue(int(width))
            spin.setMinimumHeight(30)
            spin.valueChanged.connect(
                lambda value, i=index: self._on_layer_width_changed(i, value)
            )
            layout.addWidget(spin, 1)
            layout.addWidget(label("神经元", "tiny"))
            delete_btn = icon_button("trash", "删除该层", 26, "red")
            delete_btn.setEnabled(len(config.hidden_layers) > 1)
            delete_btn.clicked.connect(lambda checked=False, i=index: self._remove_layer(i))
            layout.addWidget(delete_btn)
            self.layers_layout.addWidget(row)
            self._row_widgets.append(row)
        self.add_layer_btn.setEnabled(len(config.hidden_layers) < MAX_HIDDEN_LAYERS)

    def _on_layer_width_changed(self, index: int, value: int) -> None:
        if self._syncing:
            return
        config = self.state.model_config
        if 0 <= index < len(config.hidden_layers):
            config.hidden_layers[index] = int(value)
            self.state.notify_model_config_changed()

    def _add_layer(self) -> None:
        config = self.state.model_config
        if len(config.hidden_layers) >= MAX_HIDDEN_LAYERS:
            Toast.show_message(self, f"最多支持 {MAX_HIDDEN_LAYERS} 个隐藏层", "warning")
            return
        config.hidden_layers.append(32)
        self._refresh_layers()
        self.state.notify_model_config_changed()

    def _remove_layer(self, index: int) -> None:
        config = self.state.model_config
        if len(config.hidden_layers) <= 1:
            return
        if 0 <= index < len(config.hidden_layers):
            config.hidden_layers.pop(index)
            self._refresh_layers()
            self.state.notify_model_config_changed()

    # -- 参数回调 -----------------------------------------------------------
    def _on_activation_changed(self, index: int) -> None:
        if self._syncing or index < 0:
            return
        self.state.model_config.activation = self.activation_combo.itemData(index)
        self.state.notify_model_config_changed()

    def _on_init_changed(self, index: int) -> None:
        if self._syncing or index < 0:
            return
        self.state.model_config.init = self.init_combo.itemData(index)
        self.state.notify_model_config_changed()

    def _on_batch_norm_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self.state.model_config.batch_norm = bool(checked)
        self.state.notify_model_config_changed()

    def _on_dropout_changed(self, value: int) -> None:
        self.dropout_value.setText(f"{value}%")
        if self._syncing:
            return
        self.state.model_config.dropout = value / 100.0
        self.state.notify_model_config_changed()

    def _on_continue(self) -> None:
        if self.state.frame is None:
            Toast.show_message(self, "请先在数据页导入数据集", "error")
            return
        if self.state.prepared is None:
            try:
                self.state.prepare_current()
            except Exception as exc:  # noqa: BLE001
                Toast.show_message(self, str(exc), "error")
                return
        self.continue_requested.emit()

    def refresh_theme(self) -> None:
        self.diagram.update()
        # 刷新按钮图标
        self.back_btn.setIcon(icon("chevron-left", theme.palette.secondary, 16))
        self.add_layer_btn.setIcon(icon("plus", theme.palette.secondary, 16))


def _format_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,}"


def _format_bytes(value: int) -> str:
    size = float(max(value, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}" if size < 10 else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
