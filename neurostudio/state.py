"""应用级状态：数据集选择、预处理、模型配置与当前可用模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, Signal

from .core.data import (
    DataOptions,
    PreparedData,
    Preprocessor,
    dataframe_summary,
    detect_column_kind,
    infer_task,
    prepare_training_data,
)
from .core.model import MLP, ModelConfig, build_model
from .core.persistence import (
    build_bundle,
    bundle_summary,
    model_from_bundle,
    preprocessor_from_bundle,
)
from .core.trainer import TrainingConfig, TrainResult


@dataclass
class ActiveModel:
    model: MLP
    preprocessor: Preprocessor
    bundle: dict[str, Any]
    source: str = "本次训练"
    origin: str = "training"

    @property
    def summary(self) -> dict[str, Any]:
        return bundle_summary(self.bundle)

    @property
    def task(self) -> str:
        return self.preprocessor.task

    @property
    def label_name(self) -> str:
        return self.preprocessor.label_name


class AppState(QObject):
    data_changed = Signal()
    prepared_changed = Signal()
    model_config_changed = Signal()
    active_model_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.frame: pd.DataFrame | None = None
        self.source_path: str = ""
        self.feature_columns: list[str] = []
        self.label_column: str = ""
        self.task: str = "regression"
        self.options = DataOptions()
        self.model_config = ModelConfig(input_dim=1, hidden_layers=[64, 32], output_dim=1)
        self.training_config = TrainingConfig()
        self.prepared: PreparedData | None = None
        self.active: ActiveModel | None = None
        self.notes: dict[str, Any] = {}

    # -- 数据集 -------------------------------------------------------------
    def set_dataset(self, frame: pd.DataFrame, path: str = "") -> None:
        self.frame = frame
        self.source_path = path
        self.prepared = None
        columns = [str(c) for c in frame.columns]
        if not columns:
            raise ValueError("数据集没有列")
        self.label_column = columns[-1]
        self.task = infer_task(frame[self.label_column])
        self.feature_columns = self._default_features(frame, columns)
        self.model_config.input_dim = 1
        self.model_config.output_dim = 1
        self.data_changed.emit()

    @staticmethod
    def _default_features(frame: pd.DataFrame, columns: list[str]) -> list[str]:
        label = columns[-1]
        selected: list[str] = []
        for column in columns:
            if column == label:
                continue
            series = frame[column]
            unique = int(series.nunique(dropna=True))
            is_id = "id" in column.lower() and unique >= max(len(frame) * 0.9, 2)
            if unique <= 1 or is_id:
                continue
            selected.append(column)
        return selected or [c for c in columns if c != label]

    def invalidate(self) -> None:
        self.prepared = None

    @property
    def has_data(self) -> bool:
        return self.frame is not None and not self.frame.empty

    def source_name(self) -> str:
        return Path(self.source_path).name if self.source_path else "内置示例数据"

    def summary(self) -> dict[str, Any]:
        if self.frame is None:
            return {"rows": 0, "columns": 0, "missing": 0, "numeric_columns": 0}
        return dataframe_summary(self.frame)

    def column_kind(self, column: str) -> str:
        if self.frame is None or column not in self.frame.columns:
            return "categorical"
        return detect_column_kind(self.frame[column])

    # -- 预处理 -------------------------------------------------------------
    def prepare_current(self) -> PreparedData:
        if self.frame is None:
            raise ValueError("请先导入数据集")
        if not self.feature_columns:
            raise ValueError("请至少选择一个特征列")
        if not self.label_column:
            raise ValueError("请选择标签列")
        prepared = prepare_training_data(
            self.frame,
            self.feature_columns,
            self.label_column,
            self.task,
            self.options,
        )
        self.prepared = prepared
        self.model_config.input_dim = prepared.input_dim
        self.model_config.output_dim = prepared.output_dim
        self.model_config_changed.emit()
        self.prepared_changed.emit()
        return prepared

    def estimated_input_dim(self) -> int:
        if self.frame is None:
            return 0
        total = 0
        for column in self.feature_columns:
            if detect_column_kind(self.frame[column]) == "numeric":
                total += 1
            else:
                if self.options.categorical == "onehot":
                    total += max(int(self.frame[column].nunique(dropna=True)), 1)
                else:
                    total += 1
        return total

    # -- 模型 ---------------------------------------------------------------
    def create_model(self) -> MLP:
        if self.prepared is None:
            self.prepare_current()
        config = ModelConfig(
            input_dim=self.model_config.input_dim,
            hidden_layers=list(self.model_config.hidden_layers),
            output_dim=self.model_config.output_dim,
            activation=self.model_config.activation,
            dropout=self.model_config.dropout,
            batch_norm=self.model_config.batch_norm,
            init=self.model_config.init,
        )
        return build_model(config)

    def notify_model_config_changed(self) -> None:
        self.model_config_changed.emit()

    # -- 当前模型 -----------------------------------------------------------
    def adopt_training_result(
        self,
        result: TrainResult,
        prepared: PreparedData,
        config: TrainingConfig,
        source_name: str = "",
    ) -> ActiveModel:
        bundle = build_bundle(result, prepared, config, source_name=source_name)
        active = ActiveModel(
            model=result.model,
            preprocessor=prepared.preprocessor,
            bundle=bundle,
            source=source_name or "本次训练",
            origin="training",
        )
        self.active = active
        self.active_model_changed.emit()
        return active

    def adopt_bundle(self, bundle: dict[str, Any], source: str = "") -> ActiveModel:
        model = model_from_bundle(bundle)
        preprocessor = preprocessor_from_bundle(bundle)
        active = ActiveModel(
            model=model,
            preprocessor=preprocessor,
            bundle=bundle,
            source=source or bundle_summary(bundle).get("source") or "模型包",
            origin="bundle",
        )
        self.active = active
        self.active_model_changed.emit()
        return active
