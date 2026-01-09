# 计算神经工程 · 课程实验项目

> 本仓库为《计算神经工程》课程期末实验代码与实验流程记录。

---

## 📌 项目简介
本项目围绕 **EEG（脑电）情绪识别** 展开，基于 **DEAP 数据集** 进行分析。主要工作包括：

- **神经信号预处理**：滤波、去除伪迹、分段与基线校正
- **特征工程**：提取微分熵 (DE)、功率谱密度 (PSD) 及功能连接特征
- **深度学习建模**：构建 GCN/GAT 模型
- **评估与可视化**：Accuracy 与混淆矩阵分析

实验基于 **Python 3.10** 实现，使用 **PyTorch** 深度学习框架。

---

## 📂 项目结构
```text
LastWork/
├── data/                     # 数据存放目录
│   ├── raw/                  # 请在此放入原始 DEAP .mat 文件
│   └── processed/            # 预处理后的特征数据（由脚本自动生成）
├── src/                      # 核心代码
│   ├── data.py               # 读取数据脚本
│   ├── feature.py            # 数据处理特征工程脚本
│   ├── graph.py              # 构建图结构脚本
│   ├── preprocess.py         # 数据预处理脚本
│   ├── models/               # 模型定义文件
│   ├── train.py              # 模型训练与验证脚本
│   └── config.py             # 实验参数配置文件
├── test                      # 测试代码
├── README.md
├── requirements.txt
└── pyproject.toml
└── requirements.txt

```

## 🛠️ 使用方法
### 1. 安装依赖包
请使用 pip 安装依赖包：
```
bash
pip install -r requirements.txt
```

### 2.数据预处理
请使用 `preprocess.py` 脚本对原始数据进行预处理，生成特征数据。
运行 `python src/preprocess.py` 即可完成数据预处理。

### 3. 模型训练与验证
请使用 `train.py` 脚本对特征数据进行训练与验证。
运行 `python src/train.py` 即可完成模型训练与验证。
模型训练与验证结果将保存在 `checkpoints/` 目录下。

## 📚 参考资料
数据集获取（DEAP）
本实验使用深部脑电图数据集，需在官网申请权限后下载：
🔗数据集官网：
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
下载完成后，请求原始数据放置于本地data/目录。（也有热心网友分享，可查找）

## ⚙️参数配置说明
实验相关参数（如窗口长度、特征类型、模型超参数等）
统一在 `src/config.py` 中进行修改，不需要直接训练或废弃代码。

📎数据说明
由于EEG体积数据增大（单文件通常超过500MB），
data/ 目录未直接上传至 GitHub 仓库。

请通过DEAP官方渠道申请并下载数据后，在本地运行实验流程。
