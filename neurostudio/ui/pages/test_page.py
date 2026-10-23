"""测试与评估页：载入测试集、计算指标、可视化并导出预测结果。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.data import read_table
from ...core.metrics import classification_metrics, regression_metrics
from ...core.persistence import load_bundle
from ...core.trainer import forward_predict
from ...state import AppState
from ..theme import theme
from ..charts import ConfusionMatrixWidget, ScatterChart
from ..icons import icon
from ..help import HelpButton
from ..widgets import (
    Card,
    CardHeader,
    DataTable,
    DropZone,
    EmptyState,
    PageHeader,
    Pill,
    StatTile,
    Toast,
    label,
    primary_button,
    quiet_button,
    secondary_button,
)
from .data_page import PROJECT_ROOT


class TestPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._test_frame: pd.DataFrame | None = None
        self._test_path: str = ""
        self._export_frame: pd.DataFrame | None = None
        self._active_id: int | None = None
        self._last_dir = str(Path.home())
        self.detail_values: dict[str, QLabel] = {}
        self._build_ui()
        self.state.active_model_changed.connect(self._sync_model)
        self._sync_model(force=True)

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        header = PageHeader("评估", "在独立测试集上检验模型，并导出逐行预测结果")
        self.load_model_btn = secondary_button("加载模型包", "folder", 36)
        self.load_model_btn.clicked.connect(self._load_model)
        self.import_btn = secondary_button("导入测试集", "upload", 36)
        self.import_btn.clicked.connect(self._choose_test_file)
        header.add_action(self.load_model_btn)
        header.add_action(self.import_btn)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        page = QVBoxLayout(content)
        page.setContentsMargins(0, 0, 6, 6)
        page.setSpacing(14)

        # ---------------- 模型卡 ----------------
        model_card = Card(padding=18, spacing=10)
        self.model_pill = Pill("无模型", "gray")
        model_card.add(
            CardHeader("当前模型", "训练完成后自动载入，也可以打开导出的 ZIP / .nfs 模型包", self.model_pill)
        )
        self.model_stack = QStackedWidget()
        self.model_stack.setMinimumHeight(90)
        empty = QWidget()
        empty_layout = QHBoxLayout(empty)
        empty_layout.setContentsMargins(0, 6, 0, 6)
        empty_layout.setSpacing(14)
        empty_text = QVBoxLayout()
        empty_text.setSpacing(4)
        empty_text.addWidget(label("尚未载入模型", "card-title"))
        desc = label("先在数据页完成训练，或直接加载之前导出的模型包。", "muted")
        desc.setWordWrap(True)
        empty_text.addWidget(desc)
        empty_layout.addLayout(empty_text, 1)
        self.empty_load_btn = primary_button("加载模型包", "folder", 34)
        self.empty_load_btn.clicked.connect(self._load_model)
        empty_layout.addWidget(self.empty_load_btn)
        self.model_stack.addWidget(empty)

        model_content = QWidget()
        content_layout = QHBoxLayout(model_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(18)
        info_col = QVBoxLayout()
        info_col.setSpacing(5)
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        self.model_name = label("—", "card-title")
        name_row.addWidget(self.model_name)
        self.model_source = Pill("本次训练", "accent")
        name_row.addWidget(self.model_source)
        name_row.addStretch(1)
        info_col.addLayout(name_row)
        self.model_arch = label("—", "mono")
        info_col.addWidget(self.model_arch)
        self.model_meta = label("", "muted")
        self.model_meta.setWordWrap(True)
        info_col.addWidget(self.model_meta)
        self.model_train_meta = label("", "tiny")
        info_col.addWidget(self.model_train_meta)
        content_layout.addLayout(info_col, 3)

        score_col = QVBoxLayout()
        score_col.setSpacing(5)
        self.model_score = label("—", "value")
        self.model_score.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        score_col.addWidget(self.model_score)
        self.model_score_sub = label("训练集最终指标", "tiny")
        self.model_score_sub.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        score_col.addWidget(self.model_score_sub)
        content_layout.addLayout(score_col, 2)
        self.model_stack.addWidget(model_content)
        model_card.add(self.model_stack)
        page.addWidget(model_card)

        # ---------------- 数据 + 指标 ----------------
        middle = QHBoxLayout()
        middle.setSpacing(14)
        page.addLayout(middle)
        left = QVBoxLayout()
        left.setSpacing(14)
        right = QVBoxLayout()
        right.setSpacing(14)
        middle.addLayout(left, 5)
        middle.addLayout(right, 6)

        data_card = Card(padding=18, spacing=10)
        self.match_pill = Pill("未导入", "gray")
        data_card.add(
            CardHeader("测试数据集", "特征列名需与训练时保持一致", self.match_pill)
        )
        self.drop_zone = DropZone("拖入测试集 CSV / Excel", "或点击选择文件", "upload")
        self.drop_zone.setMinimumHeight(76)
        self.drop_zone.clicked.connect(self._choose_test_file)
        self.drop_zone.fileDropped.connect(self.load_file)
        data_card.add(self.drop_zone)
        self.test_file_name = label("尚未导入测试数据", "muted")
        self.test_file_name.setWordWrap(True)
        data_card.add(self.test_file_name)
        self.test_preview = DataTable()
        self.test_preview.setMinimumHeight(130)
        self.test_preview.load_dataframe(pd.DataFrame({"提示": ["导入测试数据后在此预览"]}))
        data_card.add(self.test_preview)
        self.match_hint = label("", "tiny")
        self.match_hint.setWordWrap(True)
        data_card.add(self.match_hint)
        evaluate_row = QHBoxLayout()
        evaluate_row.setSpacing(8)
        self.evaluate_btn = primary_button("开始评估", "play", 38)
        self.evaluate_btn.setEnabled(False)
        self.evaluate_btn.clicked.connect(self._evaluate)
        self.clear_btn = quiet_button("清除结果")
        self.clear_btn.clicked.connect(self._clear_results)
        evaluate_row.addWidget(self.evaluate_btn, 1)
        evaluate_row.addWidget(self.clear_btn)
        data_card.add_layout(evaluate_row)
        left.addWidget(data_card)

        detail_card = Card(padding=16, spacing=8)
        detail_card.add(
            CardHeader("详细指标", "补充指标与逐类报告", HelpButton("metrics"))
        )
        self.detail_grid = QGridLayout()
        self.detail_grid.setHorizontalSpacing(10)
        self.detail_grid.setVerticalSpacing(5)
        detail_card.add_layout(self.detail_grid)
        self.report_table = DataTable()
        self.report_table.setMinimumHeight(135)
        self.report_table.setVisible(False)
        detail_card.add(self.report_table)
        left.addWidget(detail_card)
        left.addStretch(1)

        self.tiles: list[StatTile] = []
        tile_row = QHBoxLayout()
        tile_row.setSpacing(8)
        for index in range(4):
            tile = StatTile("—", "—", "", "accent", "activity")
            tile.setMinimumHeight(94)
            tile_row.addWidget(tile, 1)
            self.tiles.append(tile)
        right.addLayout(tile_row)

        chart_card = Card(padding=18, spacing=8)
        self.chart_pill = Pill("等待评估", "gray")
        chart_card.add(CardHeader("评估可视化", "分类混淆矩阵 / 回归预测散点", self.chart_pill))
        self.chart_stack = QStackedWidget()
        self.chart_stack.setMinimumHeight(250)
        self.confusion = ConfusionMatrixWidget()
        self.scatter = ScatterChart()
        self.chart_stack.addWidget(self.confusion)
        self.chart_stack.addWidget(self.scatter)
        chart_card.add(self.chart_stack, 1)
        right.addWidget(chart_card, 1)

        # ---------------- 预测明细 ----------------
        pred_card = Card(padding=16, spacing=6)
        self.export_btn = secondary_button("导出预测", "download", 32)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_predictions)
        pred_card.add(
            CardHeader("预测明细", "最多显示前 200 行，导出包含全部结果", self.export_btn)
        )
        self.pred_table = DataTable()
        self.pred_table.setMinimumHeight(140)
        self.pred_table.load_dataframe(pd.DataFrame({"提示": ["尚未进行评估"]}))
        pred_card.add(self.pred_table)
        self.eval_info = label("评估完成后显示逐行预测。", "tiny")
        pred_card.add(self.eval_info)
        right.addWidget(pred_card)

        self._reset_metric_tiles("regression")
        self._refresh_empty_icon()

    def _refresh_empty_icon(self) -> None:
        pass

    def refresh_theme(self) -> None:
        self.load_model_btn.setIcon(icon("folder", theme.palette.secondary, 16))
        self.import_btn.setIcon(icon("upload", theme.palette.secondary, 16))
        self.export_btn.setIcon(icon("download", theme.palette.secondary, 15))
        self.empty_load_btn.setIcon(icon("folder", "#FFFFFF", 15))
        self.evaluate_btn.setIcon(icon("play", "#FFFFFF", 15))
        self.confusion.update()
        self.scatter.update()

    # -- 模型同步 -----------------------------------------------------------
    def _sync_model(self, force: bool = False) -> None:
        active = self.state.active
        active_id = id(active) if active is not None else None
        if not force and active_id == self._active_id:
            self._update_match_status()
            return
        self._active_id = active_id
        self._clear_results()
        if active is None:
            self.model_stack.setCurrentIndex(0)
            self.model_pill.setText("无模型")
            self.model_pill.set_tone("gray")
            self._update_match_status()
            return
        self.model_stack.setCurrentIndex(1)
        summary = active.summary
        task = active.task
        classes = summary.get("classes", [])
        self.model_pill.setText("分类模型" if task == "classification" else "回归模型")
        self.model_pill.set_tone("green" if task == "classification" else "accent")
        self.model_name.setText(Path(active.source).stem or "模型")
        is_training = active.origin == "training"
        self.model_source.setText("本次训练" if is_training else "模型包")
        self.model_source.set_tone("accent" if is_training else "purple")
        self.model_arch.setText(
            " → ".join(str(x) for x in [summary["input_dim"], *summary["config"].get("hidden_layers", []), summary["config"].get("output_dim", 1)])
        )
        label_bits = [
            f"标签 {summary['label'] or '—'}",
            f"{summary['features']} 个特征",
            f"{summary['samples']:,} 个训练样本",
        ]
        if classes:
            label_bits.append(f"{len(classes)} 个类别")
        self.model_meta.setText(" · ".join(label_bits))
        self.model_train_meta.setText(
            f"{summary['epochs']} epochs · 最佳轮次 {summary['best_epoch']} · "
            f"{_fmt_duration(summary['duration'])} · {summary['device']}"
        )
        metric_name = "accuracy" if task == "classification" else "r2"
        metric_value = summary["metrics"].get(metric_name)
        if metric_value is None:
            metric_value = summary["primary_metric"]
        if task == "classification":
            self.model_score.setText(f"{float(metric_value) * 100:.1f}%")
            self.model_score_sub.setText("训练集准确率")
        else:
            self.model_score.setText(f"{float(metric_value):.3f}")
            self.model_score_sub.setText("训练集 R²")
        self._reset_metric_tiles(task)
        self._reset_detail_metrics(task)
        self.chart_stack.setCurrentIndex(0 if task == "classification" else 1)
        self._update_match_status()

    def _reset_metric_tiles(self, task: str) -> None:
        if task == "classification":
            items = [
                ("准确率", "全部测试样本", "green", "check"),
                ("精确率", "macro 平均", "accent", "target"),
                ("召回率", "macro 平均", "orange", "trending"),
                ("F1 分数", "macro 平均", "purple", "activity"),
            ]
        else:
            items = [
                ("R²", "拟合优度，越接近 1 越好", "green", "trending"),
                ("RMSE", "均方根误差", "accent", "activity"),
                ("MAE", "平均绝对误差", "orange", "target"),
                ("MAPE", "平均绝对百分比误差", "purple", "activity"),
            ]
        for tile, (title, sub, tone, icon_name) in zip(self.tiles, items):
            tile.set_title(title)
            tile.set_tone(tone, icon_name)
            tile.set_value("—", sub)

    def _reset_detail_metrics(self, task: str) -> None:
        if not hasattr(self, "detail_grid"):
            return
        while self.detail_grid.count():
            item = self.detail_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.detail_values = {}
        if task == "classification":
            items = [
                ("roc_auc", "ROC-AUC (macro)"),
                ("mcc", "MCC"),
                ("kappa", "Cohen's κ"),
                ("log_loss", "Log loss"),
                ("specificity", "特异度 (macro)"),
                ("balanced_accuracy", "平衡准确率"),
            ]
        else:
            items = [
                ("mse", "MSE"),
                ("medae", "MedAE"),
                ("explained_variance", "解释方差"),
                ("bias", "平均偏差"),
            ]
        for index, (key, title) in enumerate(items):
            row = index // 2
            column = (index % 2) * 2
            self.detail_grid.addWidget(label(title, "muted"), row, column)
            value = label("—", "card-title")
            value.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.detail_grid.addWidget(value, row, column + 1)
            self.detail_values[key] = value
        self.detail_grid.setColumnStretch(1, 1)
        self.detail_grid.setColumnStretch(3, 1)
        self.report_table.setVisible(False)
        self.report_table.load_dataframe(pd.DataFrame({"提示": ["评估后显示分类报告"]}))

    def _update_detail_metrics(self, metrics: dict, task: str) -> None:
        if not hasattr(self, "detail_values"):
            return
        if task == "classification":
            for key in ("mcc", "kappa", "specificity", "balanced_accuracy"):
                value = metrics.get(key)
                self.detail_values[key].setText(
                    "—" if value is None else f"{float(value):.4f}"
                )
            auc = metrics.get("roc_auc")
            self.detail_values["roc_auc"].setText(
                "—" if auc is None else f"{float(auc):.4f}"
            )
            loss = metrics.get("log_loss")
            self.detail_values["log_loss"].setText(
                "—" if loss is None else f"{float(loss):.4f}"
            )
            rows = [
                {
                    "类别": item.get("name", ""),
                    "精确率": f"{item.get('precision', 0.0):.3f}",
                    "召回率": f"{item.get('recall', 0.0):.3f}",
                    "F1": f"{item.get('f1', 0.0):.3f}",
                    "支持数": str(item.get("support", 0)),
                }
                for item in metrics.get("per_class", [])
            ]
            if rows:
                self.report_table.load_dataframe(pd.DataFrame(rows))
                self.report_table.setVisible(True)
            else:
                self.report_table.setVisible(False)
        else:
            for key in ("mse", "medae", "explained_variance", "bias"):
                value = metrics.get(key)
                self.detail_values[key].setText(
                    "—" if value is None else f"{float(value):.4f}"
                )
            self.report_table.setVisible(False)

    # -- 测试数据 -----------------------------------------------------------
    def _choose_test_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入测试数据集",
            self._last_dir,
            "表格数据 (*.csv *.tsv *.tab *.txt *.xlsx *.xlsm);;所有文件 (*)",
        )
        if path:
            self._last_dir = str(Path(path).parent)
            self.load_file(path)

    def load_file(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            frame = read_table(path)
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"读取失败：{exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if frame.empty:
            Toast.show_message(self, "测试集为空", "error")
            return
        self._test_frame = frame
        self._test_path = path
        self.test_file_name.setText(f"{Path(path).name} · {len(frame):,} 行 × {frame.shape[1]} 列")
        self.test_preview.load_dataframe(frame, max_rows=60)
        self._clear_results()
        self._update_match_status()
        Toast.show_message(self, f"已导入测试集：{len(frame):,} 行", "success")

    def _update_match_status(self) -> None:
        active = self.state.active
        frame = self._test_frame
        if frame is None:
            self.match_pill.setText("未导入")
            self.match_pill.set_tone("gray")
            self.match_hint.setText("导入测试集后会自动检查列名是否匹配。")
            self.evaluate_btn.setEnabled(False)
            return
        if active is None:
            self.match_pill.setText("无模型")
            self.match_pill.set_tone("orange")
            self.match_hint.setText("测试集已就绪，但尚未载入模型。")
            self.evaluate_btn.setEnabled(False)
            return
        pre = active.preprocessor
        missing = [name for name in pre.features if name not in frame.columns]
        has_label = pre.label_name in frame.columns
        if missing:
            self.match_pill.setText(f"缺少 {len(missing)} 列")
            self.match_pill.set_tone("red")
            preview = "、".join(missing[:6]) + ("…" if len(missing) > 6 else "")
            self.match_hint.setText(f"缺少训练特征列：{preview}。请使用列名一致的数据。")
            self.evaluate_btn.setEnabled(False)
            return
        self.match_pill.setText("列匹配成功")
        self.match_pill.set_tone("green")
        hint = "已检测到标签列，可计算评价指标。" if has_label else "未检测到标签列，将仅输出预测结果。"
        self.match_hint.setText(
            f"{hint}  使用 {len(pre.features)} 个特征列，输入维度 {pre.input_dim}。"
        )
        self.evaluate_btn.setEnabled(True)

    def _clear_results(self) -> None:
        self.confusion.clear()
        self.scatter.clear()
        self.pred_table.load_dataframe(pd.DataFrame({"提示": ["尚未进行评估"]}))
        self.export_btn.setEnabled(False)
        self._export_frame = None
        self.eval_info.setText("评估完成后显示逐行预测。")
        self.chart_pill.setText("等待评估")
        self.chart_pill.set_tone("gray")
        task = self.state.active.task if self.state.active else "regression"
        self._reset_metric_tiles(task)
        self._reset_detail_metrics(task)

    # -- 评估 ---------------------------------------------------------------
    def _evaluate(self) -> None:
        active = self.state.active
        frame = self._test_frame
        if active is None:
            Toast.show_message(self, "请先载入模型", "error")
            return
        if frame is None:
            Toast.show_message(self, "请先导入测试集", "error")
            return
        pre = active.preprocessor
        missing = [name for name in pre.features if name not in frame.columns]
        if missing:
            Toast.show_message(self, f"缺少特征列：{missing[0]}", "error")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        started = time.perf_counter()
        try:
            self._ensure_device(active.model)
            X = pre.transform(frame)
            pred, probs, _ = forward_predict(active.model, X, pre.task)
            has_label = pre.label_name in frame.columns
            if has_label:
                y = pre.transform_label(frame[pre.label_name])
                if pre.task == "classification":
                    valid = y >= 0
                    metrics = classification_metrics(
                        y[valid],
                        pred[valid],
                        len(pre.classes),
                        pre.classes,
                        probabilities=np.asarray(probs)[valid],
                    )
                else:
                    valid = np.isfinite(y)
                    metrics = regression_metrics(y[valid], pred[valid])
                valid_count = int(np.asarray(valid).sum())
                if valid_count == 0:
                    has_label = False
            else:
                metrics = {}
                valid = np.zeros(len(frame), dtype=bool)
                y = np.zeros(len(frame), dtype=np.int64 if pre.task == "classification" else np.float64)
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"评估失败：{exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()

        elapsed = time.perf_counter() - started
        if has_label:
            self._apply_metrics(metrics, pre.task, int(valid.sum()))
            self._update_detail_metrics(metrics, pre.task)
            self._show_chart(metrics, pre.task, y, pred, valid if pre.task == "regression" else None)
        else:
            self._reset_metric_tiles(pre.task)
            self._reset_detail_metrics(pre.task)
            self.chart_pill.setText("仅预测")
            self.chart_pill.set_tone("orange")
            if pre.task == "classification":
                self.confusion.clear()
            else:
                self.scatter.clear()
        display, export = self._build_prediction_frames(
            frame, pre, pred, probs, y if has_label else None, valid if has_label else None
        )
        self._export_frame = export
        self.pred_table.load_dataframe(display, max_rows=200)
        self._color_result_column(display)
        self.export_btn.setEnabled(not export.empty)
        self.eval_info.setText(
            f"评估 {len(frame):,} 行 · 用时 {elapsed:.2f}s · "
            + ("已计算指标" if has_label else "测试集无标签，仅输出预测")
        )
        if has_label:
            Toast.show_message(self, "评估完成", "success")
        else:
            Toast.show_message(self, "未检测到标签列，已生成预测结果", "warning")

    @staticmethod
    def _ensure_device(model) -> None:
        if torch.cuda.is_available():
            try:
                device = next(model.parameters()).device
                if device.type == "cpu":
                    model.to("cuda")
            except Exception:  # noqa: BLE001
                pass

    def _apply_metrics(self, metrics: dict, task: str, count: int) -> None:
        if task == "classification":
            values = [
                (metrics.get("accuracy", 0.0) * 100, "accuracy"),
                (metrics.get("precision", 0.0) * 100, "precision"),
                (metrics.get("recall", 0.0) * 100, "recall"),
                (metrics.get("f1", 0.0) * 100, "f1"),
            ]
            for tile, (value, _) in zip(self.tiles, values):
                tile.set_value(f"{value:.2f}%", f"{count:,} 个有效样本")
            self.chart_pill.setText(f"混淆矩阵 · {count:,} 样本")
            self.chart_pill.set_tone("green")
        else:
            for tile, key in zip(self.tiles, ("r2", "rmse", "mae", "mape")):
                value = float(metrics.get(key, 0.0))
                if key == "mape":
                    tile.set_value(f"{value:.2f}%", f"{count:,} 个有效样本")
                elif key == "r2":
                    tile.set_value(f"{value:.4f}", f"{count:,} 个有效样本")
                else:
                    tile.set_value(f"{value:.4f}", "数值越小越好")
            self.chart_pill.setText(f"预测散点 · {count:,} 样本")
            self.chart_pill.set_tone("accent")

    def _show_chart(self, metrics: dict, task: str, y, pred, valid) -> None:
        if task == "classification":
            self.chart_stack.setCurrentIndex(0)
            labels = metrics.get("class_names") or self.state.active.preprocessor.classes
            self.confusion.set_data(metrics.get("confusion", []), labels)
        else:
            self.chart_stack.setCurrentIndex(1)
            if valid is not None:
                self.scatter.set_data(np.asarray(y)[valid], np.asarray(pred)[valid])
            else:
                self.scatter.set_data(y, pred)

    def _build_prediction_frames(self, frame, pre, pred, probs, y, valid) -> tuple[pd.DataFrame, pd.DataFrame]:
        index = np.arange(1, len(frame) + 1)
        if pre.task == "classification":
            class_names = pre.classes
            pred_labels = [
                class_names[int(p)] if 0 <= int(p) < len(class_names) else str(int(p))
                for p in pred
            ]
            confidence = (
                np.asarray(probs)[np.arange(len(pred)), pred] * 100
                if probs is not None
                else np.full(len(pred), np.nan)
            )
            if y is not None:
                true_labels = [
                    class_names[int(t)] if 0 <= int(t) < len(class_names) else "—" for t in y
                ]
                correct = [
                    ("正确" if (valid[i] and int(y[i]) == int(pred[i])) else ("错误" if valid[i] else "—"))
                    for i in range(len(pred))
                ]
            else:
                true_labels = ["—"] * len(pred)
                correct = ["—"] * len(pred)
            display = pd.DataFrame(
                {
                    "行号": index,
                    "真实标签": true_labels,
                    "预测标签": pred_labels,
                    "置信度": [f"{c:.1f}%" if np.isfinite(c) else "—" for c in confidence],
                    "结果": correct,
                }
            )
            export = display.copy()
        else:
            true_values = np.asarray(y, dtype=float) if y is not None else np.full(len(pred), np.nan)
            pred_values = np.asarray(pred, dtype=float)
            error = pred_values - true_values
            abs_error = np.abs(error)
            display = pd.DataFrame(
                {
                    "行号": index,
                    "真实值": [_fmt_float(v) for v in true_values],
                    "预测值": [_fmt_float(v) for v in pred_values],
                    "误差": [_fmt_float(v) for v in error],
                    "绝对误差": [_fmt_float(v) for v in abs_error],
                }
            )
            export = pd.DataFrame(
                {
                    "行号": index,
                    "真实值": true_values,
                    "预测值": pred_values,
                    "误差": error,
                    "绝对误差": abs_error,
                }
            )
        return display, export

    def _color_result_column(self, display: pd.DataFrame) -> None:
        if "结果" not in display.columns:
            return
        column = list(display.columns).index("结果")
        for row in range(self.pred_table.rowCount()):
            item = self.pred_table.item(row, column)
            if item is None:
                continue
            if item.text() == "错误":
                item.setForeground(QColor(theme.palette.red))
            elif item.text() == "正确":
                item.setForeground(QColor(theme.palette.green))

    # -- 导出 / 加载 --------------------------------------------------------
    def _export_predictions(self) -> None:
        frame = self._export_frame
        if frame is None or frame.empty:
            Toast.show_message(self, "没有可导出的预测结果", "warning")
            return
        stem = Path(self._test_path).stem if self._test_path else "predictions"
        default_name = f"{stem}_predictions.csv"
        start_dir = str(Path(self._test_path).parent) if self._test_path else str(PROJECT_ROOT)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出预测结果",
            str(Path(start_dir) / default_name),
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        try:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"导出失败：{exc}", "error")
            return
        Toast.show_message(self, f"预测结果已导出：{Path(path).name}", "success")

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载模型包",
            self._last_dir,
            "NeuroStudio 模型包 (*.zip *.nfs *.pt *.pth);;所有文件 (*)",
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            bundle = load_bundle(path)
            self.state.adopt_bundle(bundle, source=path)
        except Exception as exc:  # noqa: BLE001
            Toast.show_message(self, f"加载失败：{exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._last_dir = str(Path(path).parent)
        Toast.show_message(self, f"已载入模型：{Path(path).name}", "success")

    def on_show(self) -> None:
        self._update_match_status()


def _fmt_float(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    if abs(number) >= 0.01:
        return f"{number:.4f}"
    if abs(number) < 1e-12:
        return "0"
    return f"{number:.5f}"


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
