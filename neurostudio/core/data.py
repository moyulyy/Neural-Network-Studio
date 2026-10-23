"""表格数据导入、特征/标签选择与预处理流水线。

设计要点：

* 预处理统计量（缺失值填充、标准化均值/方差、类别字典）只在训练集上拟合，
  之后原样保存进模型包，保证测试数据使用完全一致的变换。
* 数值列与类别列自动判别；类别列支持 One-Hot 与序数编码。
* 不依赖 sklearn，减少运行环境的版本耦合。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ENCODINGS = ("utf-8-sig", "gbk", "utf-8", "latin-1")
TABLE_SUFFIXES = {".csv", ".tsv", ".tab", ".txt", ".xlsx", ".xlsm", ".xls"}

TASK_LABELS = {"regression": "回归", "classification": "分类"}
MISSING_LABELS = {
    "mean": "均值填充",
    "median": "中位数填充",
    "mode": "众数填充",
    "drop": "删除缺失行",
}
CATEGORICAL_LABELS = {"onehot": "One-Hot 编码", "ordinal": "序数编码"}


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def read_table(path: str | Path) -> pd.DataFrame:
    """读取 CSV / TSV / TXT / Excel，自动尝试编码与分隔符。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in TABLE_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}（支持 CSV / TSV / TXT / Excel）")

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            frame = pd.read_excel(path, sheet_name=0)
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出友好提示
            raise ValueError("读取 Excel 需要 openpyxl，请执行 pip install openpyxl") from exc
    else:
        frame = _read_delimited(path, forced_sep="\t" if suffix in {".tsv", ".tab"} else None)

    return clean_frame(frame)


def _read_delimited(path: Path, forced_sep: str | None) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        for sep in ((forced_sep,) if forced_sep else (None, ",", ";", "\t")):
            try:
                frame = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=sep,
                    engine="python" if sep is None else "c",
                    skip_blank_lines=True,
                )
                if frame.shape[1] >= 1 and len(frame) >= 0:
                    return frame
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                last_error = exc
                continue
    raise ValueError(f"无法解析数据文件：{last_error or '未知格式'}")


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """清理空行/空列、列名空白与重复列名。"""
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame = frame.reset_index(drop=True)
    seen: dict[str, int] = {}
    columns: list[str] = []
    for i, column in enumerate(frame.columns):
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed:") or name.lower() == "nan":
            name = f"列 {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        columns.append(name)
    frame.columns = columns
    return frame


# ---------------------------------------------------------------------------
# 类型推断
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _string_series(series: pd.Series) -> pd.Series:
    return series.map(_stringify)


def detect_column_kind(series: pd.Series) -> str:
    """返回 ``numeric`` 或 ``categorical``。"""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = series.dropna()
    if non_null.empty:
        return "categorical"
    converted = pd.to_numeric(non_null, errors="coerce")
    if converted.notna().mean() >= 0.95 and converted.nunique() > 2:
        return "numeric"
    return "categorical"


def infer_task(label: pd.Series) -> str:
    """根据标签列启发式判断回归 / 分类。"""
    if detect_column_kind(label) == "numeric":
        unique = int(label.dropna().nunique())
        if unique > 12 and unique > len(label) * 0.05:
            return "regression"
        return "classification"
    return "classification"


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------


@dataclass
class DataOptions:
    missing: str = "median"
    scale: bool = True
    categorical: str = "onehot"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DataOptions":
        data = data or {}
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class FeatureSpec:
    name: str
    kind: str
    encoder: str = "raw"
    fill: float = 0.0
    fill_label: str = "(缺失)"
    mean: float = 0.0
    std: float = 1.0
    categories: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class Preprocessor:
    features: list[str]
    label_name: str
    task: str
    options: DataOptions = field(default_factory=DataOptions)
    specs: list[FeatureSpec] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    input_dim: int = 0
    output_dim: int = 1

    # -- 变换 ---------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise ValueError("数据缺少特征列：" + "、".join(missing[:8]) + ("…" if len(missing) > 8 else ""))
        if frame.empty:
            raise ValueError("数据为空")
        blocks: list[np.ndarray] = []
        for spec in self.specs:
            raw = frame[spec.name]
            if spec.kind == "numeric":
                values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                values = np.where(np.isfinite(values), values, spec.fill)
                if self.options.scale and spec.encoder == "scaled":
                    values = (values - spec.mean) / (spec.std or 1.0)
                blocks.append(values.reshape(-1, 1))
            else:
                text = _string_series(raw).fillna(spec.fill_label)
                if spec.encoder == "onehot":
                    matrix = np.zeros((len(frame), len(spec.categories)), dtype=np.float64)
                    values = text.to_numpy()
                    for j, category in enumerate(spec.categories):
                        matrix[:, j] = (values == category).astype(np.float64)
                    blocks.append(matrix)
                else:
                    mapping = {name: i for i, name in enumerate(spec.categories)}
                    encoded = text.map(lambda value: mapping.get(value, -1)).to_numpy(dtype=np.float64)
                    blocks.append(encoded.reshape(-1, 1))

        matrix = (
            np.concatenate(blocks, axis=1)
            if blocks
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("特征中包含无法转换的数值，请检查数据")
        return matrix.astype(np.float32)

    def transform_label(self, series: pd.Series) -> np.ndarray:
        """返回标签数组；分类标签中缺失 / 未知类别使用 -1 标记。"""
        if self.task == "classification":
            mapping = {name: i for i, name in enumerate(self.classes)}
            text = _string_series(series)
            return text.map(lambda value: mapping.get(value, -1)).to_numpy(dtype=np.int64)
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        return values

    # -- 序列化 -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "label_name": self.label_name,
            "task": self.task,
            "options": self.options.to_dict(),
            "specs": [spec.to_dict() for spec in self.specs],
            "classes": list(self.classes),
            "feature_names": list(self.feature_names),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preprocessor":
        return cls(
            features=list(data.get("features", [])),
            label_name=str(data.get("label_name", "")),
            task=str(data.get("task", "regression")),
            options=DataOptions.from_dict(data.get("options")),
            specs=[FeatureSpec.from_dict(item) for item in data.get("specs", [])],
            classes=list(data.get("classes", [])),
            feature_names=list(data.get("feature_names", [])),
            input_dim=int(data.get("input_dim", 0)),
            output_dim=int(data.get("output_dim", 1)),
        )


@dataclass
class PreparedData:
    X: np.ndarray
    y: np.ndarray
    task: str
    preprocessor: Preprocessor
    dropped_rows: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.preprocessor.output_dim)

    @property
    def classes(self) -> list[str]:
        return list(self.preprocessor.classes)


def _fit_label(frame: pd.DataFrame, label_name: str, task: str) -> tuple[np.ndarray, list[str], pd.Series]:
    raw = frame[label_name]
    if task == "classification":
        text = _string_series(raw)
        valid = text.notna()
        if not valid.any():
            raise ValueError("标签列没有有效取值")
        classes = sorted(text[valid].unique().tolist())
        if len(classes) < 2:
            raise ValueError("分类任务至少需要两个不同类别")
        mapping = {name: i for i, name in enumerate(classes)}
        y = text.map(lambda value: mapping.get(value, -1)).to_numpy(dtype=np.int64)
        return y, classes, valid
    values = pd.to_numeric(raw, errors="coerce")
    valid = pd.Series(np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan)), index=values.index)
    if not valid.any():
        raise ValueError("回归标签必须是数值")
    y = values.to_numpy(dtype=np.float64, na_value=np.nan)
    return y, [], valid


def prepare_training_data(
    frame: pd.DataFrame,
    features: list[str],
    label_name: str,
    task: str,
    options: DataOptions | None = None,
) -> PreparedData:
    """按所选特征/标签拟合预处理并返回可直接训练的数组。"""
    options = options or DataOptions()
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("请至少选择一个特征列")
    if label_name not in frame.columns:
        raise ValueError(f"标签列不存在：{label_name}")
    if label_name in features:
        raise ValueError("标签列不能同时作为特征")
    unknown = [name for name in features if name not in frame.columns]
    if unknown:
        raise ValueError("特征列不存在：" + "、".join(unknown))
    if task not in TASK_LABELS:
        raise ValueError(f"未知任务类型：{task}")

    working = frame[features + [label_name]].copy()
    y_all, classes, label_valid = _fit_label(working, label_name, task)
    working = working.loc[label_valid].reset_index(drop=True)
    y_all = np.asarray(y_all)[label_valid.to_numpy()]

    if options.missing == "drop":
        keep = ~working[features].isna().any(axis=1)
        working = working.loc[keep].reset_index(drop=True)
        y_all = np.asarray(y_all)[keep.to_numpy()]

    if task == "regression":
        finite = np.isfinite(y_all)
        y_all = y_all[finite]
        working = working.loc[finite].reset_index(drop=True)
    if len(working) < 2:
        raise ValueError("有效样本不足（至少需要 2 行）")

    specs: list[FeatureSpec] = []
    encoded_names: list[str] = []
    for name in features:
        series = working[name]
        kind = detect_column_kind(series)
        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce").astype(np.float64)
            if options.missing == "mean":
                fill = float(values.mean())
            elif options.missing == "mode":
                modes = values.mode(dropna=True)
                fill = float(modes.iloc[0]) if not modes.empty else 0.0
            else:
                fill = float(values.median())
            if not math.isfinite(fill):
                fill = 0.0
            mean = float(values.fillna(fill).mean())
            std = float(values.fillna(fill).std(ddof=0))
            if not math.isfinite(std) or std < 1e-9:
                std = 1.0
            spec = FeatureSpec(
                name=name,
                kind="numeric",
                encoder="scaled" if options.scale else "raw",
                fill=fill,
                mean=mean,
                std=std,
                output_names=[name],
            )
        else:
            text = _string_series(series)
            valid = text.dropna()
            if valid.empty:
                raise ValueError(f"特征列「{name}」没有有效取值")
            counts = valid.value_counts()
            fill_label = str(counts.index[0])
            categories = sorted(valid.unique().tolist())
            if options.categorical == "onehot":
                spec = FeatureSpec(
                    name=name,
                    kind="categorical",
                    encoder="onehot",
                    fill_label=fill_label,
                    categories=categories,
                    output_names=[f"{name}={value}" for value in categories],
                )
            else:
                spec = FeatureSpec(
                    name=name,
                    kind="categorical",
                    encoder="ordinal",
                    fill_label=fill_label,
                    categories=categories,
                    output_names=[name],
                )
        specs.append(spec)
        encoded_names.extend(spec.output_names)

    preprocessor = Preprocessor(
        features=features,
        label_name=label_name,
        task=task,
        options=options,
        specs=specs,
        classes=classes,
        feature_names=encoded_names,
        input_dim=len(encoded_names),
        output_dim=len(classes) if task == "classification" else 1,
    )
    X = preprocessor.transform(working)
    if task == "classification":
        y = y_all.astype(np.int64)
    else:
        y = y_all.astype(np.float32)
    if len(X) != len(y):
        raise ValueError("内部错误：特征与标签行数不一致")
    dropped = max(len(frame) - len(working), 0)
    return PreparedData(X=X, y=y, task=task, preprocessor=preprocessor, dropped_rows=dropped)


def split_indices(
    y: np.ndarray, task: str, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """按比例划分训练/验证集；分类任务采用分层抽样。"""
    n = int(len(y))
    indices = np.arange(n)
    if val_ratio <= 0 or n < 4:
        return indices, np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if task == "classification":
        val_parts: list[np.ndarray] = []
        for cls in np.unique(y):
            cls_idx = indices[np.asarray(y) == cls]
            shuffled = rng.permutation(cls_idx)
            k = int(round(len(shuffled) * val_ratio))
            k = min(k, len(shuffled) - 1)
            if k > 0:
                val_parts.append(shuffled[:k].astype(np.int64))
        if not val_parts:
            return indices, np.empty(0, dtype=np.int64)
        val = np.sort(np.concatenate(val_parts))
        train = np.setdiff1d(indices, val)
        return train, val
    shuffled = rng.permutation(n)
    k = min(max(int(round(n * val_ratio)), 1), n - 1)
    return np.sort(shuffled[k:]), np.sort(shuffled[:k])


def dataframe_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing": int(frame.isna().sum().sum()),
        "numeric_columns": sum(
            1 for column in frame.columns if detect_column_kind(frame[column]) == "numeric"
        ),
    }
