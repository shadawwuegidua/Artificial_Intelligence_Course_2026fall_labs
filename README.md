# 当代人工智能：实验一

新闻文本十分类实验，代码位于 `lab1` 分支。

## 文件结构

```text
.
├── README.md
├── requirements.txt
└── 实验一/
    ├── seed_stability.py
    ├── svm_experiment.py
    ├── predict.py
    └── data-Project1/
        ├── train_data.csv
        └── test_data_unlabeled.csv
```

- `seed_stability.py`：比较朴素贝叶斯与逻辑回归的超参数，以及 TF-IDF 特征数量。
- `svm_experiment.py`：统一为 20000 个 TF-IDF 特征，比较线性核 SVC、逻辑回归与朴素贝叶斯。
- `predict.py`：在全部有标签数据上训练最终模型，生成测试集预测。

训练数据包含 `text` 和 `target` 两列，测试数据只包含 `text` 列。

## 安装环境

实验环境为 Python 3.10.22，依赖版本见 `requirements.txt`。
在仓库根目录执行：

```powershell
conda create -n ai-course-lab1 python=3.10.22
conda activate ai-course-lab1
python -m pip install -r requirements.txt
```

如果已有该环境，直接激活并安装依赖即可。

## 复现实验

在仓库根目录依次执行：

```powershell
python ".\实验一\seed_stability.py"
python ".\实验一\svm_experiment.py"
```

第一步生成 `实验一/seed_stability_results/`，第二步读取其中的同协议结果并继续实验，生成 `实验一/svm_results/`，因此需要按顺序运行。两步会输出逐轮指标、汇总表、类别指标、误分类样本及图表；SVM 实验包含多个配置和十次划分，运行时间较长。

各候选共享种子 0～9 的十次分层 80%/20% 训练与验证划分，分类器随机种子固定为 42。每次只在训练子集上拟合 TF-IDF，主要比较 Macro-F1，同时记录 Accuracy 与跨划分的样本标准差。线性核 SVC 复用精确线性核矩阵，通过 `SVC(kernel="precomputed")` 求解。

## 生成最终预测

```powershell
python ".\实验一\predict.py"
```

此步骤可以独立运行，无需先生成实验结果目录。最终方案为 `TfidfVectorizer(max_features=20000)` 与 `LogisticRegression(C=30, max_iter=1000, random_state=42)`。脚本在全部有标签数据上重新拟合词表、IDF 与分类器，按测试集原始行序生成 `实验一/prediction.csv`，格式为无表头、无索引、单列标签。

仓库仅保留复现所需的代码、输入数据和环境说明。预测文件、实验结果目录、日志、图表、检查脚本及报告草稿均保留在本地，不纳入版本控制。
