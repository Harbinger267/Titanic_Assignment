import numpy as np
import pandas as pd


def clean_train_data(input_path="../data/train.csv", output_path="../data/train_cleaned.csv"):
    train_df = pd.read_csv(input_path)

    # Missing values
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    train_df["Deck"] = train_df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "U")
    train_df = train_df.drop("Cabin", axis=1)
    train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

    # Outlier handling (Fare)
    upper_limit = train_df["Fare"].quantile(0.99)
    train_df["Fare"] = np.where(train_df["Fare"] > upper_limit, upper_limit, train_df["Fare"])
    train_df["Fare_log"] = np.log1p(train_df["Fare"])

    # Title extraction and grouping
    train_df["Title"] = train_df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    train_df["Title"] = train_df["Title"].replace(rare_titles, "Rare")

    train_df.to_csv(output_path, index=False)
    return train_df


if __name__ == "__main__":
    cleaned_df = clean_train_data()
    print("Saved cleaned dataset as train_cleaned.csv")
    print(cleaned_df.head())
