# Kaggle 泰坦尼克号：从灾难中学习机器学习

[![Kaggle](https://img.shields.io/badge/Kaggle-竞赛-blue)](https://www.kaggle.com/c/titanic)
[![最佳分数](https://img.shields.io/badge/最佳LB分数-0.79904-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()

> 利用乘客数据预测泰坦尼克号上的生存情况。经典的二分类问题，机器学习的"Hello World"。

**[English Documentation](README.md)**

---

## 排行榜演进

| 版本 | LB 分数 | 核心创新 | 经验教训 |
|:----:|:-------:|----------|----------|
| V1 | 0.75837 | 基础集成（LR, RF, GB, SVM, KNN, DT） | 默认参数，Pclass 未做 one-hot |
| V2 | 0.75837 | 仅修复 bug | 修 bug ≠ 模型提升 |
| V3 | 0.77033 | Leave-One-Out 编码 | CV 泄漏（CV 0.89 vs LB 0.77） |
| V4 | 0.77751 | 57 个特征，6 个树模型 | 过拟合（特征/样本 > 1:20） |
| V5 | 0.77272 | 单一 LGBM，保守调参 | 集成 > 单模型 |
| V6 | 0.78708 | OOF 编码 + 线性融合 | 线性融合能力有限 |
| V7 | 0.78947 | Stacking + 6 模型（5 种算法） | TicketSurvRate 泄漏 → 已修复 |
| V8 | **0.79904** | 保序校准 + 因子分析 + 分位数变换 | FN=52≫FP=32，错误聚集 |
| V9 | TBD | WCG 后处理 + 舱位侧 + 更细搜索网格 | 最新版 — 待提交 |

---

## 项目结构

```
kaggle-titanic/
├── data/                          # 原始与辅助数据
│   ├── train.csv                  # 训练集（891 行）
│   ├── test.csv                   # 测试集（418 行）
│   ├── gender_submission.csv      # Kaggle 基线
│   ├── titanic-leaked.csv         # 真实标签（验证用）
│   └── titanic-ground-true.csv    # 真实标签（备用）
├── titanic.ipynb                  # V1 — 初始 notebook
├── titanic-v3.ipynb .. v9.ipynb   # V3–V9 — 逐步改进
├── titanic-v3-executed.ipynb      # 预执行 notebook（V3）
├── titanic-v4-executed.ipynb      # 预执行 notebook（V4）
├── titanic-v5-executed.ipynb      # 预执行 notebook（V5）
├── titanic-v6-run.py              # 独立脚本（V6）
├── titanic-v7-run.py              # 独立脚本（V7）
├── _build_v7.py                   # V7 构建辅助
├── _convert.py                    # Notebook ↔ 脚本转换器
├── submission-v1.csv .. v9.csv    # Kaggle 提交文件
├── tutorial/                      # 参考教程
├── README.md                      # 英文文档
└── README.zh.md                   # 本文件
```

---

## 流水线概览

```
原始数据 → 特征工程 → OOF 目标编码 → 高级特征
    → 模型训练（6 模型 × 10 折） → 保序校准
    → 集成（Stacking + Blending） → 阈值调优 → WCG 后处理 → 提交
```

### 1. 特征工程

| 类别 | 特征 | 引入版本 |
|------|------|:--------:|
| **姓名派生** | Title（Mr/Mrs/Miss/Master/Rare）、Surname、Name_Length | V1, V8 |
| **家庭** | FamilySize、SurnameGroupSize、IsAlone、IsLargeFamily、WomanOrChild | V1 |
| **船票** | TicketPrefix、TicketGroupSize、Ticket_Frequency | V3 |
| **舱位** | Deck（ABC/DE/FG/T/U）、Cabin_num_bin（10 分位数桶）、IsStarboard | V4, V9 |
| **票价** | FarePerTicketPerson、FarePerFamilyMember、FareLog | V4 |
| **年龄** | 3 级层次化填充、AgeMissing、IsChild、AgePclass | V1 |
| **交互** | Pclass_Sex、Title_Pclass、Surname_Pclass | V7 |
| **独热编码** | Embarked、Pclass、Title、Deck、Pclass_Sex | V1+ |

**年龄填充层级**：Sex+Pclass+Title → Sex+Pclass → 全局中位数（3 级回退）

### 2. OOF 目标编码（V6+）

使用折外编码防止数据泄漏：

- **编码特征**：Title_Pclass、TicketPrefix、Surname_Pclass
- **生存率**：Surname_SurvRate、Ticket_SurvRate（仅 OOF，无全局统计）
- **配置**：贝叶斯平滑=12，StratifiedKFold 5 折

### 3. 高级特征（V8）

- **多项式交互**：degree=2, interaction_only → 互信息选择 top 10
- **分位数变换**：QuantileTransformer（uniform）应用于连续特征
- **因子分析**：2 个分量

### 4. 模型

| 模型 | 类型 | 输入特征 | 关键超参数 |
|------|------|----------|------------|
| CatBoost | 梯度提升 | 全量 | 500 轮, depth=6, lr=0.03, L2=6 |
| LightGBM | 梯度提升 | 全量 | 5000 棵, lr=0.02, leaves=64 |
| 逻辑回归 | 线性 | 稠密 | C=2.0, liblinear |
| Ridge | 线性 | 稠密 | StandardScaler + C=1.0 L2 |
| QDA | 二次判别 | 稠密 | reg_param=0.1 |
| MLP | 神经网络 | 稠密 | (100,50), adam, 早停 |

- **交叉验证**：10 折 StratifiedKFold
- **校准**：CatBoost/LGBM OOF 预测的保序回归

### 5. 集成方法

| 方法 | 描述 |
|------|------|
| **Stacking** | L1 正则化 LR 元学习器（C=0.5，嵌套 CV seed=43） |
| **平均融合** | 所有模型预测的简单均值 |
| **对数损失融合** | Dirichlet 随机搜索（15K）+ 坐标下降（6K） |

### 6. 阈值调优

- V8：0.40–0.80，步长 0.01（41 个候选）
- V9：0.35–0.75，步长 0.005（81 个候选）— 更细的网格

### 7. 后处理：WCG 规则（V9）

**妇孺组（Woman-Child-Group）** 基于家庭组生存模式覆盖预测：

- 若家庭组中所有训练成员**全部幸存** → 预测测试成员幸存
- 若家庭组中所有训练成员**全部遇难** → 预测测试成员遇难
- 灵感来自 [Chris Deotte (0.81818)](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818) 和 [Amy Peniston (81.3%)](https://www.kaggle.com/code/amypeniston/titanic-name-only-81-3)

---

## 核心经验总结

1. **CV 泄漏无声且致命** — V3 的 LOO 编码导致 CV=0.89 而 LB=0.77。务必使用 OOF 编码。
2. **更多特征 ≠ 更好** — V4 的 57 个特征在仅 891 个样本下严重过拟合。特征/样本比很关键。
3. **集成多样性胜过个体强度** — V5 的单一 LGBM 不如 V6 的融合。
4. **校准很重要** — V8 对树模型的保序校准显著提升了融合效果。
5. **领域知识缩小差距** — WCG 后处理利用了泰坦尼克号历史上"妇女儿童优先"的撤离协议。
6. **错误分析驱动改进** — V8 识别出 P3-female 和 P1-male 为系统性错误聚集，指导了 V9 的改进方向。

---

## 错误分析（V8）

| 指标 | 数值 |
|------|------|
| 假阴性（FN） | 52（漏判的幸存者） |
| 假阳性（FP） | 32（误判幸存但实际遇难） |
| FN > FP | 模型偏保守 — 低估生存率 |

**系统性错误聚集**：
- **P3-female**：三等舱女性被误分类（可能为非英语母语者，撤离通道不同）
- **P1-male**：一等舱男性被误分类（部分自愿留下或帮助他人）

---

## 快速开始

### 环境要求

```bash
pip install pandas numpy scikit-learn lightgbm catboost matplotlib seaborn missingno
```

### 运行流水线

**方式 A：Notebook**
```bash
jupyter notebook titanic-v9.ipynb
```

**方式 B：脚本（V7）**
```bash
python titanic-v7-run.py
```

### 提交结果

将生成的 `submission-v9.csv` 上传至 [Kaggle](https://www.kaggle.com/c/titanic/submissions)。

---

## 依赖项

| 包 | 用途 |
|----|------|
| pandas, numpy | 数据处理 |
| scikit-learn | 模型、交叉验证、预处理、校准 |
| lightgbm | 梯度提升（树） |
| catboost | 梯度提升（有序） |
| matplotlib, seaborn | 可视化 |
| missingno | 缺失值可视化 |

---

## 参考资料

- [Kaggle 泰坦尼克号竞赛](https://www.kaggle.com/c/titanic)
- [Chris Deotte — 仅用姓名预测（0.81818）](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818)
- [Amy Peniston — 仅用姓名预测（81.3%）](https://www.kaggle.com/code/amypeniston/titanic-name-only-81-3)
- [OOF 目标编码](https://maxhalford.github.io/blog/target-encoding/) — 防止分类编码中的数据泄漏

---

## 许可证

本项目仅供教育用途。泰坦尼克号数据集由 Kaggle 按其竞赛规则提供。
