"""训练循环、设备管理、早停与模型评估。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedData, split_indices
from .metrics import classification_metrics, regression_metrics
from .model import MLP, ModelConfig

OPTIMIZER_LABELS = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD", "rmsprop": "RMSprop"}
LOSS_LABELS = {
    "auto": "自动",
    "mse": "均方误差 (MSE)",
    "mae": "平均绝对误差 (MAE)",
    "huber": "Huber",
    "smooth_l1": "Smooth L1",
    "cross_entropy": "交叉熵",
}
SCHEDULER_LABELS = {"none": "固定学习率", "cosine": "余弦退火", "step": "阶梯衰减"}
DEVICE_LABELS = {"auto": "自动选择", "cuda": "CUDA (GPU)", "cpu": "CPU"}


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    loss: str = "auto"
    val_ratio: float = 0.2
    seed: int = 42
    device: str = "auto"
    early_stopping: int = 0
    scheduler: str = "none"
    grad_clip: float = 0.0
    class_weight: str = "none"
    label_smoothing: float = 0.0

    def validate(self) -> None:
        if not 1 <= int(self.epochs) <= 1_000_000:
            raise ValueError("训练轮数必须在 1 以上")
        if int(self.batch_size) < 1:
            raise ValueError("批大小必须至少为 1")
        if float(self.learning_rate) <= 0:
            raise ValueError("学习率必须大于 0")
        if not 0.0 <= float(self.val_ratio) < 1.0:
            raise ValueError("验证集比例必须位于 [0, 1) 区间")
        if self.optimizer not in OPTIMIZER_LABELS:
            raise ValueError(f"未知优化器：{self.optimizer}")
        if self.scheduler not in SCHEDULER_LABELS:
            raise ValueError(f"未知学习率策略：{self.scheduler}")
        if self.loss not in LOSS_LABELS:
            raise ValueError(f"未知损失函数：{self.loss}")
        if self.device not in DEVICE_LABELS:
            raise ValueError(f"未知设备：{self.device}")
        if float(self.weight_decay) < 0 or int(self.early_stopping) < 0 or float(self.grad_clip) < 0:
            raise ValueError("权重衰减 / 早停 / 梯度裁剪不能为负数")
        if self.class_weight not in {"none", "balanced"}:
            raise ValueError(f"未知类别权重策略：{self.class_weight}")
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise ValueError("标签平滑必须位于 [0, 1) 区间")

    def normalized(self) -> "TrainingConfig":
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.val_ratio = float(self.val_ratio)
        self.seed = int(self.seed)
        self.early_stopping = int(self.early_stopping)
        self.grad_clip = float(self.grad_clip)
        self.label_smoothing = float(self.label_smoothing)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in allowed}).normalized()


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境无法使用 CUDA，请检查显卡驱动与 PyTorch 安装")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"CUDA · {name}"
    return "CPU"


def primary_metric_name(task: str) -> str:
    return "accuracy" if task == "classification" else "r2"


def primary_metric_label(task: str) -> str:
    return "准确率" if task == "classification" else "R²"


def balanced_class_weights(y, n_classes: int) -> np.ndarray:
    """按类别频率计算平衡权重：n_samples / (n_present_classes * count_c)。"""
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    n_classes = max(int(n_classes), int(labels.max()) + 1 if labels.size else 0)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    if total > 0 and present.any():
        weights[present] = total / (float(present.sum()) * counts[present])
    return weights


def build_criterion(
    task: str,
    loss_name: str,
    class_weights=None,
    label_smoothing: float = 0.0,
) -> tuple[nn.Module, str]:
    if task == "classification":
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)
        return (
            nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing)),
            "cross_entropy",
        )
    name = "mse" if loss_name in {"auto", "cross_entropy"} else loss_name
    mapping: dict[str, nn.Module] = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(delta=1.0),
        "smooth_l1": nn.SmoothL1Loss(),
    }
    return mapping[name], name


class Control(Protocol):
    def should_stop(self) -> bool: ...

    def wait_if_paused(self) -> None: ...


@dataclass
class TrainResult:
    model: MLP
    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float | None
    stopped_early: bool
    duration: float
    device: str
    loss_name: str
    train_size: int
    val_size: int
    metric_name: str
    final_metrics: dict[str, Any]
    final_metrics_val: dict[str, Any]
    predictions: np.ndarray
    probabilities: np.ndarray | None
    stopped_by_user: bool = False


def _to_tensors(X: np.ndarray, y: np.ndarray, task: str) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if task == "classification":
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.long)
    else:
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    return x_tensor, y_tensor


@torch.no_grad()
def forward_predict(
    model: MLP, X: np.ndarray, task: str, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """返回 (预测值, 分类概率, 原始输出 logits)。"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    x_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    for start in range(0, len(x_tensor), batch_size):
        batch = x_tensor[start : start + batch_size].to(device)
        out = model(batch)
        logits_list.append(out.detach().cpu().numpy())
        if task == "classification":
            probs = torch.softmax(out, dim=1)
            probabilities.append(probs.detach().cpu().numpy())
            predictions.append(out.argmax(dim=1).detach().cpu().numpy())
        else:
            predictions.append(out.squeeze(-1).detach().cpu().numpy())
    model.train(was_training)
    pred = np.concatenate(predictions) if predictions else np.empty(0)
    probs = np.concatenate(probabilities) if probabilities else None
    logits = np.concatenate(logits_list) if logits_list else np.empty((0, 1))
    return pred, probs, logits


def _split_metrics(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    criterion: nn.Module,
    classes: list[str],
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray | None]:
    pred, probs, logits = forward_predict(model, X, task)
    if len(pred) == 0:
        return float("nan"), {}, pred, probs
    if task == "classification":
        weight = getattr(criterion, "weight", None)
        metric_device = (
            weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")
        )
        loss = float(
            criterion(
                torch.as_tensor(logits, dtype=torch.float32, device=metric_device),
                torch.as_tensor(y, dtype=torch.long, device=metric_device),
            ).item()
        )
        metrics = classification_metrics(y, pred, len(classes), classes)
    else:
        loss = float(criterion(torch.as_tensor(pred, dtype=torch.float32),
                               torch.as_tensor(y, dtype=torch.float32)).item())
        metrics = regression_metrics(y, pred)
    return loss, metrics, pred, probs


def train_model(
    model: MLP,
    prepared: PreparedData,
    config: TrainingConfig,
    control: Control | None = None,
    on_epoch: Callable[[dict[str, Any]], bool | None] | None = None,
    on_eval: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """训练模型并返回完整历史。``on_epoch`` 返回 False 可提前结束。"""
    config.normalized().validate()
    device = resolve_device(config.device)
    model.to(device)

    # 全局随机种子由调用方在主线程设置：在 QThread 中触发 torch.manual_seed 的
    # XPU 延迟初始化会长时间阻塞（XPU lazy call 会构造 traceback）。

    X = np.asarray(prepared.X, dtype=np.float32)
    task = prepared.task
    y = np.asarray(prepared.y)
    train_idx, val_idx = split_indices(y, task, config.val_ratio, config.seed)
    if len(train_idx) < 1:
        raise ValueError("训练集为空")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    x_train_t, y_train_t = _to_tensors(X_train, y_train, task)
    dataset = TensorDataset(x_train_t, y_train_t)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )

    class_weights = None
    if task == "classification" and config.class_weight == "balanced":
        class_weights = balanced_class_weights(y_train, len(prepared.classes))
    criterion, loss_name = build_criterion(
        task, config.loss, class_weights, config.label_smoothing
    )
    criterion = criterion.to(device)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                    momentum=0.9, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(10, config.epochs // 3), gamma=0.5
        )
    else:
        scheduler = None

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_metric": [],
        "val_metric": [],
        "lr": [],
    }
    metric_name = primary_metric_name(task)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    stopped_early = False
    stopped_by_user = False
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        if control is not None:
            control.wait_if_paused()
            if control.should_stop():
                stopped_by_user = True
                break

        model.train()
        running_loss = 0.0
        running_count = 0
        for xb, yb in loader:
            if control is not None:
                control.wait_if_paused()
                if control.should_stop():
                    stopped_by_user = True
                    break
            # BatchNorm 在单样本 batch 上无法计算方差，跳过该 batch
            if config and any(isinstance(m, nn.BatchNorm1d) for m in model.modules()) and len(xb) < 2:
                continue
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            if task == "classification":
                loss = criterion(output, yb)
            else:
                loss = criterion(output.squeeze(-1), yb)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            running_count += len(xb)

        if stopped_by_user:
            break
        if running_count == 0:
            raise ValueError("没有可用的训练批次，请增大数据量或减小批大小")

        train_loss = running_loss / running_count
        _, train_metrics, _, _ = _split_metrics(model, X_train, y_train, task, criterion, prepared.classes)
        if len(X_val) > 0:
            val_loss, val_metrics, _, _ = _split_metrics(model, X_val, y_val, task, criterion, prepared.classes)
        else:
            val_loss, val_metrics = None, {}

        if scheduler is not None:
            scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss) if val_loss is not None else float("nan"))
        history["train_metric"].append(float(train_metrics.get(metric_name, 0.0)))
        history["val_metric"].append(float(val_metrics.get(metric_name, float("nan"))))
        history["lr"].append(lr)

        monitor = val_loss if val_loss is not None else train_loss
        improved = monitor < best_loss - 1e-9
        if best_state is None or improved:
            best_loss = monitor
            best_epoch = epoch
            best_state = model.clone_state()
            bad_epochs = 0
        else:
            bad_epochs += 1

        elapsed = time.perf_counter() - started
        payload = {
            "epoch": epoch,
            "total_epochs": config.epochs,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "train_metric": float(train_metrics.get(metric_name, 0.0)),
            "val_metric": float(val_metrics.get(metric_name, float("nan"))),
            "metric_name": metric_name,
            "lr": lr,
            "elapsed": elapsed,
            "best_val_loss": float(best_loss),
            "best_epoch": best_epoch,
        }
        if on_epoch is not None and on_epoch(payload) is False:
            stopped_by_user = True
            break

        if config.early_stopping > 0 and bad_epochs >= config.early_stopping:
            stopped_early = True
            break

    duration = time.perf_counter() - started
    if stopped_by_user and len(history["epoch"]) == 0:
        raise TrainingCancelled("训练已取消")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    pred, probs, _ = forward_predict(model, X, task)
    if task == "classification":
        final_metrics = classification_metrics(
            y, pred, len(prepared.classes), prepared.classes, probabilities=probs
        )
    else:
        final_metrics = regression_metrics(y, pred)
    final_val_metrics: dict[str, Any] = {}
    if len(X_val) > 0:
        _, final_val_metrics, _, _ = _split_metrics(model, X_val, y_val, task, criterion, prepared.classes)

    if on_eval is not None:
        on_eval({"predictions": pred, "metrics": final_metrics})

    return TrainResult(
        model=model,
        history=history,
        best_epoch=int(best_epoch or len(history["epoch"])),
        best_val_loss=float(best_loss) if np.isfinite(best_loss) else None,
        stopped_early=stopped_early,
        stopped_by_user=stopped_by_user,
        duration=duration,
        device=device_description(device),
        loss_name=loss_name,
        train_size=int(len(train_idx)),
        val_size=int(len(val_idx)),
        metric_name=metric_name,
        final_metrics=final_metrics,
        final_metrics_val=final_val_metrics,
        predictions=pred,
        probabilities=probs,
    )


class TrainingCancelled(RuntimeError):
    """用户中止训练。"""


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: np.ndarray
    probabilities: np.ndarray | None
    task: str
    class_names: list[str] = field(default_factory=list)
    valid: int = 0
    total: int = 0


def evaluate_prepared(model: MLP, prepared: PreparedData) -> EvaluationResult:
    """在带标签的数据上评估。分类中标签为 -1 的行会被忽略。"""
    y = np.asarray(prepared.y)
    valid_mask = y >= 0 if prepared.task == "classification" else np.isfinite(y)
    X_valid = prepared.X[valid_mask]
    y_valid = y[valid_mask]
    pred, probs, _ = forward_predict(model, X_valid, prepared.task)
    if prepared.task == "classification":
        metrics = classification_metrics(
            y_valid,
            pred,
            len(prepared.classes),
            prepared.classes,
            probabilities=probs,
        )
    else:
        metrics = regression_metrics(y_valid, pred)
    return EvaluationResult(
        metrics=metrics,
        predictions=pred,
        probabilities=probs,
        task=prepared.task,
        class_names=list(prepared.classes),
        valid=int(valid_mask.sum()),
        total=int(len(y)),
    )
