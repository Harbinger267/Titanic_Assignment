from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "train_featured.csv"
OUTPUT_PATH = BASE_DIR / "data" / "train_selected.csv"


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "Fare_log",
        "Embarked",
        "Deck",
        "Title",
        "FamilySize",
        "IsAlone",
        "AgeGroup",
        "FarePerPerson",
    ]

    model_df = df[candidate_cols].copy()
    model_df = pd.get_dummies(
        model_df,
        columns=["Sex", "Embarked", "Deck", "Title", "AgeGroup"],
        drop_first=True,
    )
    return model_df


def select_features(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    model_df = build_model_frame(df)
    X = model_df.drop(columns="Survived")
    y = model_df["Survived"]

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)

    selected_columns = X.columns[selector.get_support()].tolist()
    selected_df = pd.concat([y, X[selected_columns]], axis=1)
    return selected_df


def main() -> None:
    featured_df = pd.read_csv(INPUT_PATH)
    selected_df = select_features(featured_df)
    selected_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved selected-feature dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
