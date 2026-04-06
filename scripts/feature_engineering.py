from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "train_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "train_featured.csv"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    featured_df = df.copy()

    featured_df["FamilySize"] = featured_df["SibSp"] + featured_df["Parch"] + 1
    featured_df["IsAlone"] = (featured_df["FamilySize"] == 1).astype(int)

    featured_df["Title"] = featured_df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )

    featured_df["AgeGroup"] = pd.cut(
        featured_df["Age"],
        bins=[0, 12, 19, 59, np.inf],
        labels=["Child", "Teen", "Adult", "Senior"],
        include_lowest=True,
    )

    featured_df["TicketPrefix"] = (
        featured_df["Ticket"]
        .astype(str)
        .str.replace(r"\d+", "", regex=True)
        .str.replace(r"[./]", "", regex=True)
        .str.replace(" ", "", regex=False)
        .replace("", "NUM")
    )

    featured_df["FarePerPerson"] = featured_df["Fare"] / featured_df["FamilySize"]

    return featured_df


def main() -> None:
    train_df = pd.read_csv(INPUT_PATH)
    featured_df = add_engineered_features(train_df)
    featured_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved engineered dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
