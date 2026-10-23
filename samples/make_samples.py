"""生成 NeuroStudio 的示例数据集（只需在开发时运行一次）。

依赖：numpy、pandas；iris 使用 scikit-learn 内置数据（如未安装则跳过）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261022)


def make_housing(n: int = 900) -> pd.DataFrame:
    area = RNG.normal(92, 32, n).clip(28, 260)
    age = RNG.integers(0, 36, n)
    rooms = RNG.integers(1, 6, n)
    floor = RNG.integers(1, 33, n)
    distance = RNG.gamma(2.0, 1.6, n).clip(0.3, 22)
    school = RNG.normal(7.8, 1.3, n).clip(3.5, 10)
    green = RNG.normal(0.34, 0.11, n).clip(0.05, 0.8)
    decoration = RNG.choice(["精装", "简装", "毛坯"], n, p=[0.32, 0.5, 0.18])
    orientation = RNG.choice(["南", "南北", "东", "西", "北"], n, p=[0.34, 0.3, 0.16, 0.12, 0.08])
    decorate_bonus = pd.Series(decoration).map({"精装": 22.0, "简装": 0.0, "毛坯": -12.0}).to_numpy()
    orient_bonus = pd.Series(orientation).map({"南": 8.0, "南北": 10.0, "东": 0.0, "西": -4.0, "北": -7.0}).to_numpy()
    price = (
        1.85 * area
        - 0.42 * age
        + 3.4 * rooms
        + 0.28 * floor
        - 1.15 * distance
        + 4.6 * school
        + 32.0 * green
        + decorate_bonus
        + orient_bonus
        + RNG.normal(0, 9.5, n)
    ).clip(18, None)
    return pd.DataFrame(
        {
            "面积": area.round(1),
            "房龄": age,
            "卧室数": rooms,
            "楼层": floor,
            "距地铁_km": distance.round(2),
            "学区评分": school.round(2),
            "绿化率": green.round(3),
            "装修": decoration,
            "朝向": orientation,
            "成交价_万": price.round(2),
        }
    )


def make_churn(n: int = 1400) -> pd.DataFrame:
    tenure = RNG.integers(1, 73, n)
    monthly = RNG.normal(78, 24, n).clip(18, 160)
    support = RNG.poisson(1.4, n)
    age = RNG.integers(18, 76, n)
    usage = RNG.normal(28, 9, n).clip(2, 65)
    contract = RNG.choice(["月付", "一年", "两年"], n, p=[0.5, 0.3, 0.2])
    internet = RNG.choice(["光纤", "DSL", "无"], n, p=[0.55, 0.32, 0.13])
    autopay = RNG.choice(["是", "否"], n, p=[0.62, 0.38])
    contract_risk = pd.Series(contract).map({"月付": 1.5, "一年": -0.2, "两年": -1.2}).to_numpy()
    internet_risk = pd.Series(internet).map({"光纤": 0.35, "DSL": -0.25, "无": 0.5}).to_numpy()
    autopay_risk = pd.Series(autopay).map({"是": -0.55, "否": 0.45}).to_numpy()
    logit = (
        -1.35
        - 0.045 * tenure
        + 0.017 * monthly
        + 0.58 * support
        - 0.035 * usage
        + 0.012 * (age - 40)
        + contract_risk
        + internet_risk
        + autopay_risk
        + RNG.normal(0, 0.65, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, probability)
    return pd.DataFrame(
        {
            "在网月数": tenure,
            "月费": monthly.round(2),
            "客服呼叫次数": support,
            "年龄": age,
            "月均流量_GB": usage.round(1),
            "合约类型": contract,
            "接入方式": internet,
            "自动扣款": autopay,
            "是否流失": churn,
        }
    )


def make_iris() -> pd.DataFrame | None:
    try:
        from sklearn.datasets import load_iris
    except ImportError:
        return None
    data = load_iris(as_frame=True)
    frame = data.frame.copy()
    frame.columns = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度", "品种"]
    names = {0: "山鸢尾", 1: "变色鸢尾", 2: "维吉尼亚鸢尾"}
    frame["品种"] = frame["品种"].map(names)
    return frame


def main() -> None:
    housing = make_housing()
    housing.to_csv(ROOT / "housing.csv", index=False, encoding="utf-8-sig")
    print(f"housing.csv: {housing.shape}")
    churn = make_churn()
    churn.to_csv(ROOT / "churn.csv", index=False, encoding="utf-8-sig")
    print(f"churn.csv: {churn.shape}")
    iris = make_iris()
    if iris is not None:
        iris.to_csv(ROOT / "iris.csv", index=False, encoding="utf-8-sig")
        print(f"iris.csv: {iris.shape}")
    else:
        print("跳过 iris.csv（未安装 scikit-learn）")


if __name__ == "__main__":
    main()
