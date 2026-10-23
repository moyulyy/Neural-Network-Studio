"""核心功能测试：动态结构、预处理、指标、训练与模型包往返。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neurostudio.core.data import (
    DataOptions,
    detect_column_kind,
    infer_task,
    prepare_training_data,
    read_table,
    split_indices,
)
from neurostudio.core.metrics import classification_metrics, regression_metrics
from neurostudio.core.model import ModelConfig, build_model, estimate_model_size
from neurostudio.core.persistence import (
    build_bundle,
    load_bundle,
    model_from_bundle,
    preprocessor_from_bundle,
    save_bundle,
)
from neurostudio.core.trainer import (
    TrainingConfig,
    balanced_class_weights,
    build_criterion,
    evaluate_prepared,
    forward_predict,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 动态网络
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden, output_dim, batch_norm, dropout",
    [
        ([16], 1, False, 0.0),
        ([32, 16, 8], 3, True, 0.1),
        ([8, 8], 2, False, 0.25),
    ],
)
def test_dynamic_architecture(hidden, output_dim, batch_norm, dropout):
    config = ModelConfig(
        input_dim=6,
        hidden_layers=hidden,
        output_dim=output_dim,
        activation="gelu",
        dropout=dropout,
        batch_norm=batch_norm,
        init="xavier_uniform",
    )
    model = build_model(config)
    output = model(torch.randn(12, 6))
    assert output.shape == (12, output_dim)
    assert model.parameter_count > 0

    extra = ModelConfig.from_dict(config.to_dict())
    assert extra.hidden_layers == hidden
    assert extra.output_dim == output_dim


def test_estimate_model_size():
    stats = estimate_model_size(
        ModelConfig(
            input_dim=5,
            hidden_layers=[7, 3],
            output_dim=2,
            batch_norm=True,
        )
    )
    assert stats.weight_count == 72
    assert stats.bias_count == 22
    assert stats.parameter_count == 94
    assert stats.fp32_bytes == 94 * 4
    assert stats.fp16_bytes == 94 * 2
    assert stats.activation_elements == 5 + 7 + 3 + 2
    model = build_model(
        ModelConfig(input_dim=5, hidden_layers=[7, 3], output_dim=2, batch_norm=True)
    )
    assert model.parameter_count == stats.parameter_count


def test_model_rejects_empty_hidden_layers():
    with pytest.raises(ValueError):
        build_model(ModelConfig(input_dim=4, hidden_layers=[], output_dim=1))


def test_parameter_count_matches_manual_formula():
    model = build_model(
        ModelConfig(input_dim=5, hidden_layers=[7, 3], output_dim=2, batch_norm=True)
    )
    expected = (5 * 7 + 7) + (7 * 3 + 3) + (3 * 2 + 2) + 2 * (7 + 3)
    assert model.parameter_count == expected


# ---------------------------------------------------------------------------
# 数据预处理
# ---------------------------------------------------------------------------


def test_infer_task_and_kind():
    rng = np.random.default_rng(0)
    continuous = pd.Series(rng.normal(size=200))
    classes = pd.Series(np.tile(["甲", "乙", "丙"], 40))
    assert infer_task(continuous) == "regression"
    assert infer_task(classes) == "classification"
    assert detect_column_kind(pd.Series([1, 2, 3.5])) == "numeric"
    assert detect_column_kind(pd.Series(["a", "b", "a"])) == "categorical"
    assert detect_column_kind(pd.Series([True, False])) == "categorical"


def test_prepare_with_missing_and_categorical():
    frame = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "num2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "cat": ["a", "b", "a", None, "b", "a", "b", "a"],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    prepared = prepare_training_data(
        frame,
        ["num1", "num2", "cat"],
        "price",
        "regression",
        DataOptions(missing="mean", scale=True, categorical="onehot"),
    )
    assert prepared.input_dim == 4  # num1 + num2 + cat(a) + cat(b)
    assert prepared.X.shape == (8, 4)
    assert np.isfinite(prepared.X).all()
    assert prepared.y.shape == (8,)
    assert prepared.preprocessor.feature_names[:2] == ["num1", "num2"]

    test_frame = pd.DataFrame({"num1": [3.0], "num2": [1.0], "cat": ["never-seen"]})
    transformed = prepared.preprocessor.transform(test_frame)
    assert transformed.shape == (1, 4)
    assert transformed[0, 2] == 0 and transformed[0, 3] == 0


def test_prepare_drop_rows_keeps_labels_aligned():
    frame = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]}
    )
    prepared = prepare_training_data(
        frame, ["x"], "y", "regression", DataOptions(missing="drop", scale=False)
    )
    assert prepared.n_samples == 3
    assert np.allclose(prepared.y, [10.0, 30.0, 40.0])
    assert np.allclose(prepared.X[:, 0], [1.0, 3.0, 4.0])


def test_classification_encoding_and_roundtrip():
    frame = pd.read_csv(ROOT / "samples" / "iris.csv")
    features = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
    prepared = prepare_training_data(frame, features, "品种", "classification")
    assert prepared.input_dim == 4
    assert prepared.output_dim == 3
    assert set(prepared.classes) == {"山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"}
    assert set(np.unique(prepared.y)) == {0, 1, 2}

    frame2 = pd.DataFrame(
        {
            "kind": ["a", "b", "c", "a"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "label": ["x", "y", "x", "y"],
        }
    )
    pre2 = prepare_training_data(
        frame2,
        ["kind", "value"],
        "label",
        "classification",
        DataOptions(categorical="ordinal", scale=False),
    )
    assert pre2.input_dim == 2
    assert pre2.preprocessor.specs[0].categories == ["a", "b", "c"]


def test_split_indices_stratified():
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    train, val = split_indices(y, "classification", 0.25, 7)
    assert len(train) + len(val) == len(y)
    assert set(y[val]) == {0, 1, 2}
    assert len(val) == 15


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def test_regression_metrics():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["medae"] == pytest.approx(0.0)
    assert metrics["explained_variance"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.0)
    metrics = regression_metrics([1, 2, 3], [2, 2, 3])
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["medae"] == pytest.approx(0.0)


def test_classification_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2, ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion"] == [[1, 1], [0, 2]]
    assert metrics["class_names"] == ["a", "b"]
    assert metrics["per_class"][1]["recall"] == pytest.approx(1.0)
    assert 0.0 <= metrics["mcc"] <= 1.0
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_probability_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.92, 0.05, 0.03],
            [0.88, 0.08, 0.04],
            [0.04, 0.90, 0.06],
            [0.06, 0.86, 0.08],
            [0.02, 0.06, 0.92],
            [0.05, 0.05, 0.90],
        ]
    )
    metrics = classification_metrics(
        y, y, 3, ["a", "b", "c"], probabilities=probabilities
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["kappa"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["log_loss"] is not None and metrics["log_loss"] < 0.2


def test_binary_auc_known_value():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    y = np.array([0, 0, 1, 1])
    metrics = classification_metrics(
        y, np.array([0, 0, 0, 1]), 2, ["neg", "pos"], probabilities=probabilities
    )
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_balanced_class_weights_and_criterion():
    weights = balanced_class_weights([0, 0, 0, 1], 2)
    assert np.allclose(weights, [4 / (2 * 3), 4 / (2 * 1)])
    criterion, name = build_criterion(
        "classification", "auto", class_weights=weights, label_smoothing=0.1
    )
    assert name == "cross_entropy"
    assert criterion.weight is not None
    assert np.allclose(criterion.weight.detach().numpy(), weights)
    assert criterion.label_smoothing == pytest.approx(0.1)


def test_training_config_validation_for_imbalance_options():
    config = TrainingConfig(class_weight="balanced", label_smoothing=0.1)
    config.validate()
    restored = TrainingConfig.from_dict(config.to_dict())
    assert restored.class_weight == "balanced"
    assert restored.label_smoothing == pytest.approx(0.1)
    with pytest.raises(ValueError):
        TrainingConfig(class_weight="unknown").validate()
    with pytest.raises(ValueError):
        TrainingConfig(label_smoothing=1.0).validate()


# ---------------------------------------------------------------------------
# 训练 / 持久化
# ---------------------------------------------------------------------------


def _regression_frame(n: int = 240, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.6, -2.2, 0.7]) + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["target"] = y
    return frame


def test_train_regression_and_bundle_roundtrip(tmp_path):
    frame = _regression_frame()
    prepared = prepare_training_data(
        frame,
        ["a", "b", "c"],
        "target",
        "regression",
        DataOptions(missing="median", scale=True),
    )
    torch.manual_seed(11)
    model = build_model(ModelConfig(input_dim=3, hidden_layers=[16, 8], output_dim=1))
    config = TrainingConfig(
        epochs=70,
        batch_size=16,
        learning_rate=0.01,
        val_ratio=0.2,
        seed=11,
        device="cpu",
        optimizer="adam",
        scheduler="cosine",
    )
    result = train_model(model, prepared, config)
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]
    assert result.final_metrics["r2"] > 0.85
    assert result.best_epoch >= 1
    assert len(result.history["epoch"]) == 70

    evaluation = evaluate_prepared(result.model, prepared)
    assert evaluation.metrics["rmse"] < 1.0

    bundle = build_bundle(result, prepared, config, source_name="synthetic.csv")
    assert bundle["model_config"]["hidden_layers"] == [16, 8]
    assert bundle["training"]["history"]["train_loss"]
    # 新版 ZIP：weights.pt + model.json
    zip_path = save_bundle(bundle, tmp_path / "model.zip")
    assert zip_path.exists() and zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert {"weights.pt", "model.json"} <= names
        meta = json.loads(archive.read("model.json").decode("utf-8"))
    assert meta["weights_file"] == "weights.pt"
    assert meta["model_config"]["hidden_layers"] == [16, 8]
    assert meta["dataset"]["label"] == "target"
    assert len(meta["training"]["history"]["epoch"]) == 70
    assert meta["training"]["final_metrics"]["r2"] > 0.85

    loaded = load_bundle(zip_path)
    restored = model_from_bundle(loaded)
    pre = preprocessor_from_bundle(loaded)
    X = pre.transform(frame[["a", "b", "c"]])
    predictions, _, _ = forward_predict(restored, X, "regression")
    assert np.allclose(predictions, result.predictions, atol=1e-5)
    assert loaded["dataset"]["label"] == "target"

    # 旧版 .nfs 仍可载入
    legacy_path = save_bundle(bundle, tmp_path / "model.nfs")
    legacy = load_bundle(legacy_path)
    assert model_from_bundle(legacy) is not None


def test_train_classification_without_validation(tmp_path):
    frame = pd.read_csv(ROOT / "samples" / "iris.csv")
    features = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
    prepared = prepare_training_data(frame, features, "品种", "classification")
    torch.manual_seed(5)
    model = build_model(ModelConfig(input_dim=4, hidden_layers=[32, 16], output_dim=3))
    config = TrainingConfig(
        epochs=90,
        batch_size=16,
        learning_rate=0.01,
        val_ratio=0.0,
        seed=5,
        device="cpu",
        early_stopping=0,
    )
    result = train_model(model, prepared, config)
    assert result.final_metrics["accuracy"] > 0.9
    assert result.val_size == 0
    assert np.isnan(result.history["val_loss"][0])
    assert np.isnan(result.history["val_metric"][0])
    assert len(result.final_metrics["confusion"]) == 3

    bundle = build_bundle(result, prepared, config, source_name="iris.csv")
    zip_path = save_bundle(bundle, tmp_path / "iris.zip")
    with zipfile.ZipFile(zip_path) as archive:
        meta = json.loads(archive.read("model.json").decode("utf-8"))
    # NaN 必须被转换成标准 JSON 的 null
    assert meta["training"]["history"]["val_loss"][0] is None
    assert meta["training"]["history"]["val_metric"][0] is None
    loaded = load_bundle(zip_path)
    restored = model_from_bundle(loaded)
    pre = preprocessor_from_bundle(loaded)
    predictions, probabilities, _ = forward_predict(
        restored, pre.transform(frame[features]), "classification"
    )
    assert predictions.shape[0] == 150
    assert probabilities is not None and probabilities.shape == (150, 3)


def test_early_stopping_runs():
    frame = _regression_frame(160)
    prepared = prepare_training_data(frame, ["a", "b", "c"], "target", "regression")
    torch.manual_seed(1)
    model = build_model(ModelConfig(input_dim=3, hidden_layers=[8], output_dim=1))
    config = TrainingConfig(
        epochs=500,
        batch_size=32,
        learning_rate=0.05,
        val_ratio=0.2,
        seed=1,
        device="cpu",
        early_stopping=3,
    )
    result = train_model(model, prepared, config)
    assert result.best_epoch >= 1
    assert len(result.history["epoch"]) <= 500
    assert result.stopped_early or len(result.history["epoch"]) == 500


def test_read_table_samples():
    frame = read_table(ROOT / "samples" / "housing.csv")
    assert frame.shape[0] == 900
    assert frame.shape[1] == 10
    assert "成交价_万" in frame.columns


def test_cuda_status_report():
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")


def test_imbalanced_training_with_class_weights_and_smoothing():
    rng = np.random.default_rng(9)
    n_major, n_minor = 240, 30
    major = rng.normal(loc=-0.2, scale=1.0, size=(n_major, 3))
    minor = rng.normal(loc=2.2, scale=1.0, size=(n_minor, 3))
    X = np.vstack([major, minor])
    y = np.array([0] * n_major + [1] * n_minor)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["target"] = y
    prepared = prepare_training_data(frame, ["a", "b", "c"], "target", "classification")
    torch.manual_seed(9)
    model = build_model(ModelConfig(input_dim=3, hidden_layers=[16], output_dim=2))
    config = TrainingConfig(
        epochs=35,
        batch_size=32,
        learning_rate=0.01,
        val_ratio=0.2,
        seed=3,
        device="cpu",
        class_weight="balanced",
        label_smoothing=0.05,
    )
    result = train_model(model, prepared, config)
    assert result.final_metrics["count"] == n_major + n_minor
    assert result.final_metrics["accuracy"] > 0.85
    assert len(result.history["epoch"]) == 35
