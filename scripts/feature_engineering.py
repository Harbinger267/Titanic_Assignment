import pandas as pd


def engineer_features(input_path="../data/train_cleaned.csv", output_path="../data/train_featured.csv"):
    df = pd.read_csv(input_path)

    # Family-based features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Standardize title variants
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Age banding
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 19, 59, float("inf")],
        labels=["Child", "Teen", "Adult", "Senior"],
        include_lowest=True,
    )

    # Extract ticket prefix
    df["TicketPrefix"] = (
        df["Ticket"]
        .astype(str)
        .str.replace(r"\d+", "", regex=True)
        .str.replace(r"[./]", "", regex=True)
        .str.replace(" ", "", regex=False)
        .replace("", "NUM")
    )

    # Fare normalized by family size
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    engineered_df = engineer_features()
    print("Saved engineered dataset as train_featured.csv")
    print(engineered_df.head())
