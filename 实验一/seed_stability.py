"""用十次分层随机划分复核实验 1 的模型、参数和词表大小结论。

在 ai-course-lab1 环境运行：python seed_stability.py
保持原 main.py 与其结果不变；本脚本输出到 seed_stability_results/。
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data-Project1" / "train_data.csv"
OUTPUT_DIR = BASE_DIR / "seed_stability_results"
SPLIT_SEEDS = tuple(range(10))
MODEL_SEED = 42
VALIDATION_SIZE = 0.2


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    parameter: str
    value: float
    max_features: int


EXPERIMENTS = (
    Experiment("NB_a0.1_f5000", "MultinomialNB", "alpha", 0.1, 5000),
    Experiment("NB_a1_f5000", "MultinomialNB", "alpha", 1.0, 5000),
    Experiment("NB_a10_f5000", "MultinomialNB", "alpha", 10.0, 5000),
    Experiment("LR_C0.1_f5000", "LogisticRegression", "C", 0.1, 5000),
    Experiment("LR_C1_f5000", "LogisticRegression", "C", 1.0, 5000),
    Experiment("LR_C10_f5000", "LogisticRegression", "C", 10.0, 5000),
    Experiment("LR_C30_f5000", "LogisticRegression", "C", 30.0, 5000),
    Experiment("LR_C100_f5000", "LogisticRegression", "C", 100.0, 5000),
    Experiment("LR_C100_f10000", "LogisticRegression", "C", 100.0, 10000),
    Experiment("LR_C100_f20000", "LogisticRegression", "C", 100.0, 20000),
)

# 正差表示左侧方案更好。预先列出对照，避免看完结果再挑有利比较。
COMPARISONS = (
    ("LR_C1_f5000", "NB_a1_f5000"),
    ("LR_C100_f5000", "NB_a0.1_f5000"),
    ("NB_a0.1_f5000", "NB_a1_f5000"),
    ("NB_a1_f5000", "NB_a10_f5000"),
    ("LR_C1_f5000", "LR_C0.1_f5000"),
    ("LR_C10_f5000", "LR_C1_f5000"),
    ("LR_C30_f5000", "LR_C10_f5000"),
    ("LR_C100_f5000", "LR_C30_f5000"),
    ("LR_C100_f10000", "LR_C100_f5000"),
    ("LR_C100_f20000", "LR_C100_f10000"),
)
ANALYSIS_EXPERIMENT = "LR_C100_f20000"


def make_classifier(experiment):
    if experiment.model == "MultinomialNB":
        return MultinomialNB(alpha=experiment.value)
    return LogisticRegression(
        C=experiment.value,
        max_iter=1000,
        random_state=MODEL_SEED,
    )


def run_experiments(data):
    """同一种子下共享划分；同一词表上限只拟合一次 TF-IDF。"""
    rows = []
    class_rows = []
    prediction_frames = []
    labels = np.sort(data["target"].unique())

    for seed in SPLIT_SEEDS:
        round_started = perf_counter()
        train_indices, val_indices = train_test_split(
            data.index.to_numpy(),
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=data["target"],
        )
        train_text = data.loc[train_indices, "text"]
        val_text = data.loc[val_indices, "text"]
        y_train = data.loc[train_indices, "target"]
        y_val = data.loc[val_indices, "target"]

        for feature_limit in (5000, 10000, 20000):
            feature_started = perf_counter()
            vectorizer = TfidfVectorizer(max_features=feature_limit)
            # 词表和 IDF 只从当前轮的训练子集学习。
            train_features = vectorizer.fit_transform(train_text)
            val_features = vectorizer.transform(val_text)
            feature_seconds = perf_counter() - feature_started

            for experiment in EXPERIMENTS:
                if experiment.max_features != feature_limit:
                    continue
                classifier = make_classifier(experiment)
                fit_started = perf_counter()
                classifier.fit(train_features, y_train)
                fit_seconds = perf_counter() - fit_started
                predictions = classifier.predict(val_features)
                rows.append({
                    "seed": seed,
                    **asdict(experiment),
                    "n_train": len(train_indices),
                    "n_val": len(val_indices),
                    "actual_features": train_features.shape[1],
                    "accuracy": accuracy_score(y_val, predictions),
                    "macro_f1": f1_score(y_val, predictions, average="macro"),
                    "feature_seconds": feature_seconds,
                    "fit_seconds": fit_seconds,
                })

                if experiment.name == ANALYSIS_EXPERIMENT:
                    precision, recall, f1, support = precision_recall_fscore_support(
                        y_val, predictions, labels=labels,
                    )
                    for index, label in enumerate(labels):
                        class_rows.append({
                            "seed": seed,
                            "label": int(label),
                            "precision": precision[index],
                            "recall": recall[index],
                            "f1": f1[index],
                            "support": int(support[index]),
                        })
                    prediction_frames.append(pd.DataFrame({
                        "seed": seed,
                        "source_row": val_indices,
                        "true_label": y_val.to_numpy(),
                        "predicted_label": predictions,
                    }))
                    print(
                        f"seed={seed:2d} | {experiment.name} | "
                        f"Accuracy={rows[-1]['accuracy']:.4f} | "
                        f"Macro-F1={rows[-1]['macro_f1']:.4f} | "
                        f"round={perf_counter() - round_started:.1f}s",
                        flush=True,
                    )

        # 每轮保存一次已完成的实验，运行日志中也能定位当前进度。
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "runs.csv", index=False)

    return (
        pd.DataFrame(rows),
        pd.DataFrame(class_rows),
        pd.concat(prediction_frames, ignore_index=True),
        labels,
    )


def summarize_runs(runs):
    grouping = ["name", "model", "parameter", "value", "max_features"]
    summary = runs.groupby(grouping, as_index=False).agg(
        n_splits=("seed", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"),
        macro_f1_max=("macro_f1", "max"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    ranks = runs.groupby("seed")["macro_f1"].rank(
        ascending=False, method="min",
    )
    winners = runs.assign(is_best=ranks.eq(1)).groupby("name")["is_best"].sum()
    summary["best_split_count"] = summary["name"].map(winners)
    summary = summary.sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    return summary


def paired_comparisons(runs):
    """每次先在同一划分上求差，再汇总差异，而非比较无关划分。"""
    accuracy = runs.pivot(index="seed", columns="name", values="accuracy")
    macro_f1 = runs.pivot(index="seed", columns="name", values="macro_f1")
    difference_frames = []
    rows = []
    for left, right in COMPARISONS:
        accuracy_delta = accuracy[left] - accuracy[right]
        f1_delta = macro_f1[left] - macro_f1[right]
        difference_frames.append(pd.DataFrame({
            "seed": macro_f1.index,
            "left": left,
            "right": right,
            "accuracy_delta": accuracy_delta.to_numpy(),
            "macro_f1_delta": f1_delta.to_numpy(),
        }))
        rows.append({
            "left": left,
            "right": right,
            "n_splits": len(f1_delta),
            "accuracy_delta_mean": accuracy_delta.mean(),
            "macro_f1_delta_mean": f1_delta.mean(),
            "macro_f1_delta_std": f1_delta.std(),
            "macro_f1_delta_min": f1_delta.min(),
            "macro_f1_delta_max": f1_delta.max(),
            "f1_wins": int((f1_delta > 0).sum()),
            "f1_ties": int((f1_delta == 0).sum()),
            "f1_losses": int((f1_delta < 0).sum()),
        })
    paired = pd.DataFrame(rows)
    paired.to_csv(OUTPUT_DIR / "paired_summary.csv", index=False)
    pd.concat(difference_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "paired_differences.csv", index=False,
    )
    return paired


def summarize_errors(data, class_metrics, predictions, labels):
    class_metrics.to_csv(OUTPUT_DIR / "class_metrics.csv", index=False)
    class_summary = class_metrics.groupby("label", as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    ).sort_values("f1_mean")
    class_summary.to_csv(OUTPUT_DIR / "class_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    errors = predictions.loc[
        predictions["true_label"] != predictions["predicted_label"]
    ].copy()
    errors["text"] = errors["source_row"].map(data["text"])
    errors.to_csv(
        OUTPUT_DIR / "validation_errors.csv", index=False, encoding="utf-8-sig",
    )
    error_pairs = errors.groupby(
        ["true_label", "predicted_label"], as_index=False,
    ).agg(count=("seed", "size"), splits_present=("seed", "nunique"))
    error_pairs = error_pairs.sort_values("count", ascending=False)
    error_pairs.to_csv(OUTPUT_DIR / "error_pairs.csv", index=False)

    # 同一新闻可能在多个验证集中出现；该矩阵统计预测次数而非独立新闻数。
    matrix = confusion_matrix(
        predictions["true_label"], predictions["predicted_label"], labels=labels,
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "confusion_matrix_sum.csv", index_label="true_label",
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set_title("LR C=100, 20000 features\nSummed predictions: 10 overlapping splits")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_sum.png", dpi=160)
    plt.close(fig)
    return class_summary, error_pairs


def plot_stability(runs, summary):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    studies = (
        ("LogisticRegression", 5000, "value", "C", "LR regularization"),
        ("MultinomialNB", 5000, "value", "alpha", "Naive Bayes smoothing"),
        ("LogisticRegression", None, "max_features", "Max features", "LR C=100 vocabulary"),
    )
    for ax, (model, feature_limit, x_column, x_label, title) in zip(axes, studies):
        selected = runs[runs["model"].eq(model)]
        selected_summary = summary[summary["model"].eq(model)]
        if feature_limit is None:
            selected = selected[selected["value"].eq(100.0)]
            selected_summary = selected_summary[selected_summary["value"].eq(100.0)]
        else:
            selected = selected[selected["max_features"].eq(feature_limit)]
            selected_summary = selected_summary[
                selected_summary["max_features"].eq(feature_limit)
            ]
        for _, seed_rows in selected.groupby("seed"):
            seed_rows = seed_rows.sort_values(x_column)
            ax.plot(
                seed_rows[x_column], seed_rows["macro_f1"],
                color="0.75", alpha=0.65, linewidth=0.9,
            )
        selected_summary = selected_summary.sort_values(x_column)
        ax.errorbar(
            selected_summary[x_column], selected_summary["macro_f1_mean"],
            yerr=selected_summary["macro_f1_std"],
            color="#165D9C", marker="o", capsize=4, linewidth=2,
            label="Mean +/- sample SD",
        )
        if x_column == "value":
            ax.set_xscale("log")
        else:
            ax.set_xticks([5000, 10000, 20000], ["5000", "10000", "20000"])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Validation Macro-F1")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stability.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    metadata = {
        "split_seeds": SPLIT_SEEDS,
        "model_seed": MODEL_SEED,
        "validation_size": VALIDATION_SIZE,
        "split_method": "Repeated stratified 80/20 holdout; shared splits per configuration",
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "data_rows": len(data),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "interpretation": (
            "Seed 42 was used for development and is excluded from this repeat. "
            "SD is sample standard deviation across overlapping validation splits, "
            "not a standard error or confidence interval. These are validation "
            "results on the same labeled dataset, not an independent test score. "
            "Timing is descriptive and feature construction time is shared."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(
        f"Running {len(SPLIT_SEEDS)} new splits x {len(EXPERIMENTS)} configurations; "
        "model random_state remains 42.",
        flush=True,
    )
    runs, class_metrics, predictions, labels = run_experiments(data)
    summary = summarize_runs(runs)
    paired = paired_comparisons(runs)
    class_summary, error_pairs = summarize_errors(
        data, class_metrics, predictions, labels,
    )
    plot_stability(runs, summary)
    print("\n=== Mean and sample SD over 10 new splits ===")
    print(summary[[
        "name", "accuracy_mean", "accuracy_std", "macro_f1_mean",
        "macro_f1_std", "best_split_count",
    ]].to_string(index=False, float_format="%.6f"))
    print("\n=== Paired changes: left minus right; wins use Macro-F1 ===")
    print(paired[[
        "left", "right", "accuracy_delta_mean", "macro_f1_delta_mean",
        "f1_wins", "f1_ties", "f1_losses",
    ]].to_string(index=False, float_format="%.6f"))
    print("\n=== LR C=100, 20000 features: per-class stability ===")
    print(class_summary.to_string(index=False, float_format="%.6f"))
    print("\n=== Most common error directions (counts may repeat documents) ===")
    print(error_pairs.head(8).to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
