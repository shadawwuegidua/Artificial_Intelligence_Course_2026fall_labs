"""在同样的十次分层划分与两万维 TF-IDF 上比较 SVM、LR 和 NB。

运行：python svm_experiment.py
SVM 使用课程的线性 SVC；精确线性核矩阵每轮计算一次，供不同 C 共用。
通过 SVC(kernel="precomputed") 求解，仍是一对一线性核 SVC，不是 LinearSVC。
已有的同协议 LR C=100 结果会复用，所有新输出位于 svm_results/。
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter, sleep

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
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel

from seed_stability import (
    DATA_PATH,
    MODEL_SEED,
    SPLIT_SEEDS,
    VALIDATION_SIZE,
    Experiment,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "svm_results"
PREVIOUS_DIR = BASE_DIR / "seed_stability_results"
MAX_FEATURES = 20000
EXPERIMENTS = (
    *(Experiment(f"SVM_C{c:g}_f20000", "SVC", "C", c, MAX_FEATURES)
      for c in (0.1, 1.0, 10.0, 100.0)),
    *(Experiment(f"LR_C{c:g}_f20000", "LogisticRegression", "C", c, MAX_FEATURES)
      for c in (10.0, 30.0, 100.0)),
    *(Experiment(f"NB_a{alpha:g}_f20000", "MultinomialNB", "alpha", alpha, MAX_FEATURES)
      for alpha in (0.1, 1.0, 10.0)),
)
PREDICTION_COLUMNS = ["name", "seed", "source_row", "true_label", "predicted_label"]


def make_classifier(experiment):
    if experiment.model == "SVC":
        return SVC(
            kernel="precomputed", C=experiment.value,
            cache_size=512, random_state=MODEL_SEED,
        )
    if experiment.model == "LogisticRegression":
        return LogisticRegression(
            C=experiment.value, max_iter=1000, random_state=MODEL_SEED,
        )
    return MultinomialNB(alpha=experiment.value)


def save_csv(frame, path):
    """先写临时文件，再替换正式结果，避免中断留下半个 CSV。"""
    temporary_path = path.with_suffix(".tmp")
    frame.to_csv(temporary_path, index=False)
    # Windows 的文件监视器/扫描程序可能短暂占用刚保存的 CSV。
    for attempt in range(8):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            sleep(0.05 * 2 ** attempt)


def prepare_results(protocol):
    """保留逐配置检查点；只复用数据和协议相同的已完成实验。"""
    metadata_path = OUTPUT_DIR / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["protocol"] != protocol:
            raise ValueError("svm_results 中已有不同协议的结果，请使用另一个输出目录。")
        if (OUTPUT_DIR / "runs.csv").exists():
            runs = pd.read_csv(OUTPUT_DIR / "runs.csv").drop_duplicates(
                ["seed", "name"], keep="last",
            )
            predictions = pd.read_csv(OUTPUT_DIR / "validation_predictions.csv")
            # 预测先保存，指标后提交；恢复时丢弃未提交配置的孤立预测。
            predictions = predictions.merge(runs[["seed", "name"]], on=["seed", "name"])
            predictions = predictions.drop_duplicates(["name", "seed", "source_row"])
            return runs, predictions

    old_metadata = json.loads(
        (PREVIOUS_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        old_metadata["data_sha256"] != protocol["data_sha256"]
        or old_metadata["model_seed"] != MODEL_SEED
        or old_metadata["validation_size"] != VALIDATION_SIZE
        or old_metadata["versions"] != protocol["versions"]
    ):
        raise ValueError("先前结果的数据、环境或划分设置已变化，不能直接复用。")
    old_runs = pd.read_csv(PREVIOUS_DIR / "runs.csv")
    reused = old_runs.loc[
        old_runs["name"].eq("LR_C100_f20000")
        & old_runs["seed"].isin(SPLIT_SEEDS)
    ].copy()
    old_predictions = pd.read_csv(PREVIOUS_DIR / "validation_predictions.csv")
    predictions = old_predictions.loc[old_predictions["seed"].isin(SPLIT_SEEDS)].copy()
    predictions["name"] = "LR_C100_f20000"
    return reused, predictions[PREDICTION_COLUMNS]


def run_experiments(data, runs, predictions):
    rows = runs.to_dict("records")
    prediction_frames = [predictions]
    completed = set(zip(runs["seed"], runs["name"]))
    reference_path = OUTPUT_DIR / "direct_linear_reference_predictions.csv"
    reference = pd.read_csv(reference_path) if reference_path.exists() else pd.DataFrame()
    equivalence_path = OUTPUT_DIR / "kernel_equivalence.csv"
    equivalence_rows = (
        pd.read_csv(equivalence_path).to_dict("records")
        if equivalence_path.exists() else []
    )
    for seed in SPLIT_SEEDS:
        pending = [experiment for experiment in EXPERIMENTS
                   if (seed, experiment.name) not in completed]
        if not pending:
            print(f"seed={seed}: all configurations already saved", flush=True)
            continue
        train_indices, val_indices = train_test_split(
            data.index.to_numpy(), test_size=VALIDATION_SIZE,
            random_state=seed, stratify=data["target"],
        )
        y_train = data.loc[train_indices, "target"]
        y_val = data.loc[val_indices, "target"]
        feature_started = perf_counter()
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
        train_features = vectorizer.fit_transform(data.loc[train_indices, "text"])
        val_features = vectorizer.transform(data.loc[val_indices, "text"])
        feature_seconds = perf_counter() - feature_started
        kernel_seconds = 0.0
        if any(experiment.model == "SVC" for experiment in pending):
            kernel_started = perf_counter()
            # K_train[i,j] = <x_train_i, x_train_j>，即精确线性核。
            train_kernel = linear_kernel(train_features)
            val_kernel = linear_kernel(val_features, train_features)
            kernel_seconds = perf_counter() - kernel_started
            print(f"seed={seed:2d} | exact linear Gram matrices: {kernel_seconds:.1f}s", flush=True)

        for experiment in pending:
            print(f"seed={seed:2d} | fitting {experiment.name} ...", flush=True)
            classifier = make_classifier(experiment)
            fit_features = train_kernel if experiment.model == "SVC" else train_features
            predict_features = val_kernel if experiment.model == "SVC" else val_features
            fit_started = perf_counter()
            classifier.fit(fit_features, y_train)
            fit_seconds = perf_counter() - fit_started
            predicted = classifier.predict(predict_features)
            rows.append({
                "seed": seed,
                **asdict(experiment),
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "actual_features": train_features.shape[1],
                "accuracy": accuracy_score(y_val, predicted),
                "macro_f1": f1_score(y_val, predicted, average="macro"),
                "feature_seconds": feature_seconds,
                "kernel_seconds": kernel_seconds if experiment.model == "SVC" else 0.0,
                "fit_seconds": fit_seconds,
            })
            prediction_frames.append(pd.DataFrame({
                "name": experiment.name,
                "seed": seed,
                "source_row": val_indices,
                "true_label": y_val.to_numpy(),
                "predicted_label": predicted,
            }))
            if not reference.empty:
                direct = reference.loc[
                    reference["seed"].eq(seed) & reference["name"].eq(experiment.name)
                ]
                if not direct.empty:
                    matched = prediction_frames[-1].merge(
                        direct, on=["name", "seed", "source_row"],
                        suffixes=("_precomputed", "_direct"), validate="one_to_one",
                    )
                    different = int((
                        matched["predicted_label_precomputed"]
                        != matched["predicted_label_direct"]
                    ).sum())
                    if len(matched) != len(val_indices) or different:
                        raise ValueError("精确预计算线性核与直接线性 SVC 的预测不一致。")
                    equivalence_rows.append({
                        "seed": seed, "name": experiment.name,
                        "n_compared": len(matched), "n_different": different,
                    })
                    save_csv(pd.DataFrame(equivalence_rows), equivalence_path)
                    print(f"  direct linear SVC agreement: {len(matched)}/{len(matched)}", flush=True)
            # 核 SVM 耗时较长，每个配置完成后保存，重启时不重复拟合。
            save_csv(
                pd.concat(prediction_frames, ignore_index=True),
                OUTPUT_DIR / "validation_predictions.csv",
            )
            save_csv(pd.DataFrame(rows), OUTPUT_DIR / "runs.csv")
            print(
                f"seed={seed:2d} | {experiment.name} | "
                f"Accuracy={rows[-1]['accuracy']:.4f} | "
                f"Macro-F1={rows[-1]['macro_f1']:.4f} | fit={fit_seconds:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def summarize(runs):
    summary = runs.groupby(
        ["name", "model", "parameter", "value", "max_features"], as_index=False,
    ).agg(
        n_splits=("seed", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"),
        macro_f1_max=("macro_f1", "max"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    best_by_model = summary.groupby("model", sort=False).head(1)
    best_by_model.to_csv(OUTPUT_DIR / "best_by_model.csv", index=False)
    svm_name = best_by_model.loc[best_by_model["model"].eq("SVC"), "name"].iloc[0]
    comparisons = [
        (f"SVM_C{left:g}_f20000", f"SVM_C{right:g}_f20000")
        for left, right in ((1, 0.1), (10, 1), (100, 10))
    ]
    comparisons.extend((
        ("LR_C10_f20000", "LR_C100_f20000"),
        ("LR_C30_f20000", "LR_C100_f20000"),
    ))
    comparisons.extend(
        (svm_name, row["name"])
        for _, row in best_by_model.iterrows() if row["model"] != "SVC"
    )
    f1 = runs.pivot(index="seed", columns="name", values="macro_f1")
    accuracy = runs.pivot(index="seed", columns="name", values="accuracy")
    paired_rows = []
    differences = []
    for left, right in comparisons:
        delta = f1[left] - f1[right]
        accuracy_delta = accuracy[left] - accuracy[right]
        paired_rows.append({
            "left": left, "right": right,
            "accuracy_delta_mean": accuracy_delta.mean(),
            "macro_f1_delta_mean": delta.mean(),
            "macro_f1_delta_std": delta.std(),
            "f1_wins": int((delta > 0).sum()),
            "f1_ties": int((delta == 0).sum()),
            "f1_losses": int((delta < 0).sum()),
        })
        differences.append(pd.DataFrame({
            "seed": f1.index, "left": left, "right": right,
            "accuracy_delta": accuracy_delta.to_numpy(),
            "macro_f1_delta": delta.to_numpy(),
        }))
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT_DIR / "paired_summary.csv", index=False)
    pd.concat(differences, ignore_index=True).to_csv(
        OUTPUT_DIR / "paired_differences.csv", index=False,
    )
    return summary, best_by_model, paired


def analyze_predictions(data, predictions, best_by_model):
    labels = np.sort(data["target"].unique())
    class_rows = []
    error_frames = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (_, experiment) in zip(axes, best_by_model.iterrows()):
        selected = predictions.loc[predictions["name"].eq(experiment["name"])]
        for seed, seed_predictions in selected.groupby("seed"):
            precision, recall, f1, support = precision_recall_fscore_support(
                seed_predictions["true_label"], seed_predictions["predicted_label"],
                labels=labels,
            )
            for index, label in enumerate(labels):
                class_rows.append({
                    "name": experiment["name"], "model": experiment["model"],
                    "seed": seed, "label": int(label),
                    "precision": precision[index], "recall": recall[index],
                    "f1": f1[index], "support": int(support[index]),
                })
        matrix = confusion_matrix(
            selected["true_label"], selected["predicted_label"], labels=labels,
        )
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d",
            text_kw={"fontsize": 7},
        )
        ax.set_title(experiment["name"])
        errors = selected.loc[
            selected["true_label"] != selected["predicted_label"]
        ].copy()
        errors["text"] = errors["source_row"].map(data["text"])
        error_frames.append(errors)
    fig.suptitle("Best candidate per model: summed predictions over 10 overlapping splits")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    class_metrics = pd.DataFrame(class_rows)
    class_metrics.to_csv(OUTPUT_DIR / "class_metrics.csv", index=False)
    class_metrics.groupby(["name", "model", "label"], as_index=False).agg(
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    ).to_csv(OUTPUT_DIR / "class_summary.csv", index=False)
    pd.concat(error_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_errors.csv", index=False, encoding="utf-8-sig",
    )


def plot_comparison(runs, summary, best_by_model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    svm_runs = runs.loc[runs["model"].eq("SVC")]
    for _, seed_rows in svm_runs.groupby("seed"):
        seed_rows = seed_rows.sort_values("value")
        axes[0].plot(seed_rows["value"], seed_rows["macro_f1"], color="0.75", alpha=0.7)
    svm_summary = summary.loc[summary["model"].eq("SVC")].sort_values("value")
    axes[0].errorbar(
        svm_summary["value"], svm_summary["macro_f1_mean"],
        yerr=svm_summary["macro_f1_std"], marker="o", capsize=4,
        color="#165D9C", label="Mean +/- sample SD",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("Validation Macro-F1")
    axes[0].set_title("SVC with exact linear kernel, 20000 TF-IDF features")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    x = np.arange(len(best_by_model))
    axes[1].errorbar(
        x, best_by_model["macro_f1_mean"], yerr=best_by_model["macro_f1_std"],
        fmt="o", markersize=7, capsize=5, color="#165D9C",
    )
    for _, seed_rows in runs.groupby("seed"):
        scores = seed_rows.set_index("name").loc[best_by_model["name"], "macro_f1"]
        axes[1].scatter(x, scores, color="0.6", alpha=0.5, s=16)
    axes[1].set_xticks(x, best_by_model["name"], rotation=12, ha="right")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].set_title("Best candidate per model, same 10 splits and features")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    protocol = {
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "split_seeds": list(SPLIT_SEEDS), "validation_size": VALIDATION_SIZE,
        "model_seed": MODEL_SEED, "max_features": MAX_FEATURES,
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "svm": {
            "kernel": "linear", "execution": "precomputed_exact_linear_gram",
            "cache_size": 512,
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    runs, predictions = prepare_results(protocol)
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps({
        "protocol": protocol,
        "reused_source": "../seed_stability_results/runs.csv: LR_C100_f20000",
        "interpretation": (
            "Repeated stratified 80/20 validation; all models use identical features "
            "within a split. SD is descriptive variation, not a confidence interval. "
            "Best candidates are selected on these same validation splits; their "
            "comparison is exploratory, not an unbiased independent test. "
            "Summed confusion matrices count repeated documents across splits. "
            "SVC uses an exact linear Gram matrix shared across C values; "
            "kernel construction is timed separately from fitting. "
            "Existing LR C=100 timings were recorded in an earlier run."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{len(SPLIT_SEEDS)} splits x {len(EXPERIMENTS)} configurations; "
        f"reusing {len(runs)} completed fits. All use 20000 TF-IDF features.",
        flush=True,
    )
    runs, predictions = run_experiments(data, runs, predictions)
    summary, best_by_model, paired = summarize(runs)
    analyze_predictions(data, predictions, best_by_model)
    plot_comparison(runs, summary, best_by_model)
    print("\n=== All configurations, mean +/- sample SD ===")
    print(summary.to_string(index=False, float_format="%.6f"))
    print("\n=== Paired differences: left minus right ===")
    print(paired.to_string(index=False, float_format="%.6f"))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
