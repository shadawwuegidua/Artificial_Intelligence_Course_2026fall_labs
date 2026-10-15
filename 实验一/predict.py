"""用全部有标签数据训练最终模型，生成提交用的 prediction.csv。"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data-Project1"
    train_df = pd.read_csv(data_dir / "train_data.csv")
    test_df = pd.read_csv(data_dir / "test_data_unlabeled.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("classifier", LogisticRegression(
            C=30.0,
            max_iter=1000,
            random_state=42,
        )),
    ])

    # 最终方案已通过验证集选定，这里使用全部有标签数据重新拟合。
    model.fit(train_df["text"], train_df["target"])
    # 保持测试集原始行序；predict 沿用训练时学到的词表与 IDF。
    predictions = model.predict(test_df["text"])

    output_path = base_dir / "prediction.csv"
    pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)
    print(f"最终训练样本数：{len(train_df)}")
    print(f"测试集预测行数：{len(predictions)}")
    print(f"已生成：{output_path}")


if __name__ == "__main__":
    main()
