"""可完全自定义的多层感知机（MLP）。

输入维度、隐藏层数量、每层神经元数、输出维度均由 :class:`ModelConfig`
描述。模型同时可以被序列化成普通 dict，方便写入模型包。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
}

ACTIVATION_LABELS: dict[str, str] = {
    "relu": "ReLU",
    "leaky_relu": "LeakyReLU",
    "gelu": "GELU",
    "silu": "SiLU / Swish",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "elu": "ELU",
    "selu": "SELU",
}

INITIALIZERS: dict[str, str] = {
    "kaiming_uniform": "Kaiming 均匀",
    "kaiming_normal": "Kaiming 正态",
    "xavier_uniform": "Xavier 均匀",
    "xavier_normal": "Xavier 正态",
    "normal": "正态 (σ=0.02)",
}

MAX_HIDDEN_LAYERS = 16


@dataclass
class ModelConfig:
    """网络结构定义。"""

    input_dim: int = 8
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 1
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    init: str = "kaiming_uniform"

    def validate(self) -> None:
        if self.input_dim < 1:
            raise ValueError("输入维度必须至少为 1")
        if self.output_dim < 1:
            raise ValueError("输出维度必须至少为 1")
        if not self.hidden_layers:
            raise ValueError("至少需要一个隐藏层")
        if len(self.hidden_layers) > MAX_HIDDEN_LAYERS:
            raise ValueError(f"隐藏层最多 {MAX_HIDDEN_LAYERS} 层")
        if any(int(n) < 1 for n in self.hidden_layers):
            raise ValueError("每层神经元数量必须至少为 1")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"未知激活函数: {self.activation}")
        if self.init not in INITIALIZERS:
            raise ValueError(f"未知初始化方式: {self.init}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("Dropout 必须位于 [0, 1) 区间")

    def normalized(self) -> "ModelConfig":
        """返回一份字段类型正确的副本。"""
        self.input_dim = int(self.input_dim)
        self.hidden_layers = [int(n) for n in self.hidden_layers]
        self.output_dim = int(self.output_dim)
        self.dropout = float(self.dropout)
        self.batch_norm = bool(self.batch_norm)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed}).normalized()

    def describe(self) -> str:
        layers = [self.input_dim, *self.hidden_layers, self.output_dim]
        return " → ".join(str(n) for n in layers)


class MLP(nn.Module):
    """由 :class:`ModelConfig` 生成的顺序多层感知机。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.normalized().validate()
        self.config = config

        layers: list[nn.Module] = []
        activation_cls = ACTIVATIONS[config.activation]
        prev = config.input_dim
        for width in config.hidden_layers:
            layers.append(nn.Linear(prev, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation_cls())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev = width
        layers.append(nn.Linear(prev, config.output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Linear):
            return
        fan_in = module.weight.shape[1]
        init = self.config.init
        if init == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        elif init == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:  # pragma: no cover - validate() 已拦截
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            bound = 1.0 / math.sqrt(max(fan_in, 1))
            nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_specs(self) -> list[dict[str, Any]]:
        """供网络结构图使用的逐层信息。"""
        specs: list[dict[str, Any]] = [
            {"kind": "input", "label": "输入", "width": self.config.input_dim}
        ]
        for i, width in enumerate(self.config.hidden_layers, start=1):
            specs.append(
                {
                    "kind": "hidden",
                    "label": f"隐藏 {i}",
                    "width": width,
                    "activation": ACTIVATION_LABELS.get(
                        self.config.activation, self.config.activation
                    ),
                    "dropout": self.config.dropout,
                    "batch_norm": self.config.batch_norm,
                }
            )
        specs.append({"kind": "output", "label": "输出", "width": self.config.output_dim})
        return specs

    def clone_state(self) -> dict[str, torch.Tensor]:
        """CPU 上的权重副本（用于早停恢复与保存）。"""
        return {
            key: value.detach().to("cpu").clone()
            for key, value in self.state_dict().items()
        }


def build_model(config: ModelConfig) -> MLP:
    return MLP(config)
