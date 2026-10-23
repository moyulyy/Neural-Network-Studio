"""模型包的构建、保存与载入。

新版模型包是一个 ZIP 文件，包含：

* ``weights.pt`` —— PyTorch ``state_dict`` 权重
* ``model.json`` —— 网络结构、预处理参数、训练参数、完整曲线、最佳轮次、
  最终指标与数据集摘要（人类可读）

旧版 ``.nfs``（torch.save 的整包字典）仍然可以载入。
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import PreparedData, Preprocessor
from .model import MLP, ModelConfig
from .trainer import TrainingConfig, TrainResult

FORMAT_NAME = "neurostudio-model"
FORMAT_VERSION = 1
APP_VERSION = "1.0.0"

META_NAME = "model.json"
WEIGHTS_NAME = "weights.pt"


def build_bundle(
    result: TrainResult,
    prepared: PreparedData,
    config: TrainingConfig,
    source_name: str = "",
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pre = prepared.preprocessor
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "app": {"name": "NeuroStudio", "version": APP_VERSION},
        "model_config": result.model.config.to_dict(),
        "preprocessor": pre.to_dict(),
        "state_dict": result.model.clone_state(),
        "training": {
            "config": config.to_dict(),
            "history": result.history,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "stopped_by_user": result.stopped_by_user,
            "duration_seconds": result.duration,
            "device": result.device,
            "loss_name": result.loss_name,
            "metric_name": result.metric_name,
            "train_size": result.train_size,
            "val_size": result.val_size,
            "final_metrics": result.final_metrics,
            "final_metrics_val": result.final_metrics_val,
        },
        "dataset": {
            "source": source_name,
            "rows": int(prepared.n_samples),
            "dropped_rows": int(prepared.dropped_rows),
            "task": prepared.task,
            "features": list(pre.features),
            "encoded_features": list(pre.feature_names),
            "label": pre.label_name,
            "classes": list(pre.classes),
            "input_dim": int(pre.input_dim),
            "output_dim": int(pre.output_dim),
        },
        "metrics": extra_metrics or {},
    }


# ---------------------------------------------------------------------------
# JSON 序列化
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """把 bundle 转换成严格合法 JSON 可接受的结构（NaN/Inf → null）。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def bundle_to_json(bundle: dict[str, Any]) -> dict[str, Any]:
    """生成写入 ``model.json`` 的元数据（不含权重张量）。"""
    payload = {key: value for key, value in bundle.items() if key != "state_dict"}
    payload["weights_file"] = WEIGHTS_NAME
    payload["weights_format"] = "pytorch state_dict"
    return _json_safe(payload)


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


def _save_zip_bundle(bundle: dict[str, Any], path: Path) -> Path:
    if "state_dict" not in bundle:
        raise ValueError("模型包缺少权重 state_dict")
    state = {
        str(key): value.detach().to("cpu").clone()
        for key, value in bundle["state_dict"].items()
    }
    payload = bundle_to_json(bundle)
    meta_text = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False, default=str
    )
    with tempfile.TemporaryDirectory(prefix="neurostudio_export_") as tmp:
        weights_path = Path(tmp) / WEIGHTS_NAME
        torch.save(state, weights_path)
        with zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.write(weights_path, WEIGHTS_NAME)
            archive.writestr(META_NAME, meta_text)
    return path


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """保存模型包。``.zip`` 输出 ZIP（推荐）；``.nfs/.pt`` 输出旧版格式。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".zip", ".nfs", ".pt", ".pth"}:
        path = path.with_suffix(".zip")
        suffix = ".zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".zip":
        return _save_zip_bundle(bundle, path)
    torch.save(bundle, path)
    return path


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------


def _read_zip_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if META_NAME not in names:
            raise ValueError(f"ZIP 模型包缺少 {META_NAME}")
        payload = json.loads(archive.read(META_NAME).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("model.json 格式不正确")
        weights_name = str(payload.get("weights_file") or WEIGHTS_NAME)
        if weights_name not in names:
            raise ValueError(f"ZIP 模型包缺少权重文件 {weights_name}")
        weights_bytes = archive.read(weights_name)
    state = torch.load(
        io.BytesIO(weights_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(state, dict):
        raise ValueError("权重文件格式不正确")
    payload["state_dict"] = state
    return payload


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    bundle: dict[str, Any] | None = None

    # 同时兼容新版 ZIP 与旧版 torch.save 整包
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            names = set()
        if META_NAME in names:
            bundle = _read_zip_bundle(path)
        elif path.suffix.lower() == ".zip":
            raise ValueError(f"ZIP 模型包缺少 {META_NAME}")

    if bundle is None:
        bundle = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(bundle, dict):
        raise ValueError("模型文件格式不正确")
    if bundle.get("format") != FORMAT_NAME:
        raise ValueError("这不是 NeuroStudio 模型包（缺少格式标记）")
    if "state_dict" not in bundle or "model_config" not in bundle or "preprocessor" not in bundle:
        raise ValueError("模型包缺少必要字段")
    if int(bundle.get("format_version", 0)) > FORMAT_VERSION:
        raise ValueError("模型包由更新版本的 NeuroStudio 创建，请升级后再试")
    return bundle


def model_from_bundle(bundle: dict[str, Any]) -> MLP:
    config = ModelConfig.from_dict(bundle["model_config"])
    model = MLP(config)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def preprocessor_from_bundle(bundle: dict[str, Any]) -> Preprocessor:
    return Preprocessor.from_dict(bundle["preprocessor"])


def bundle_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    config = bundle.get("model_config", {})
    training = bundle.get("training", {})
    dataset = bundle.get("dataset", {})
    metrics = training.get("final_metrics", {})
    task = dataset.get("task", "regression")
    primary = "accuracy" if task == "classification" else "r2"
    return {
        "created_at": bundle.get("created_at", ""),
        "architecture": " → ".join(
            str(n)
            for n in [config.get("input_dim", 0), *config.get("hidden_layers", []),
                      config.get("output_dim", 0)]
        ),
        "task": task,
        "task_label": "分类" if task == "classification" else "回归",
        "label": dataset.get("label", ""),
        "features": len(dataset.get("features", [])),
        "input_dim": int(dataset.get("input_dim", 0)),
        "samples": int(dataset.get("rows", 0)),
        "classes": list(dataset.get("classes", [])),
        "epochs": len(training.get("history", {}).get("epoch", [])),
        "best_epoch": training.get("best_epoch", 0),
        "duration": float(training.get("duration_seconds", 0.0) or 0.0),
        "primary_metric": float(metrics.get(primary, 0.0) or 0.0),
        "primary_name": primary,
        "metrics": metrics,
        "device": training.get("device", ""),
        "loss_name": training.get("loss_name", ""),
        "source": dataset.get("source", ""),
        "config": config,
        "training_config": training.get("config", {}),
        "history": training.get("history", {}),
    }
