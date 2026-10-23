# NeuroStudio 快速预测窗

一个仅用于推理的小窗口，不包含数据导入和训练功能。

## 启动

```bash
conda activate neurostudio
python tests/app/predict_app.py
```

也可以直接传入模型包路径：

```bash
python tests/app/predict_app.py path/to/model.zip
```

或双击 `run_predict.bat`（可把 ZIP 拖到 bat 上）。

## 使用

1. 加载训练页导出的 ZIP 模型包（也兼容旧版 `.nfs`）。
2. 窗口会根据 `model.json` 中的特征信息自动生成输入框：
   - 数值特征：预填训练集均值，可直接修改；
   - 类别特征：从训练集中的类别中选择。
3. 点击「开始预测」（或在数值框按 Enter）。
   - 回归：显示推理出的数值；
   - 分类：显示预测类别、置信度与各类概率。

## 模型包结构

```
model.zip
├─ weights.pt     # PyTorch state_dict
└─ model.json     # 网络结构、预处理、训练参数/曲线/指标、数据集摘要
```

权重只在本地通过 `torch.load(..., weights_only=True)` 加载，请使用可信来源的模型包。
