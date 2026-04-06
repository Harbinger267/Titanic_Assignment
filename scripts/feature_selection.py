import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def select_features(input_path="../data/train_featured.csv", output_path="../data/train_selected.csv", k=10):
    df = pd.read_csv(input_path)

    model_df = df[
        [
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
    ].copy()

    model_df = pd.get_dummies(
        model_df,
        columns=["Sex", "Embarked", "Deck", "Title", "AgeGroup"],
        drop_first=True,
    )

    X = model_df.drop("Survived", axis=1)
    y = model_df["Survived"]

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)

    selected_features = X.columns[selector.get_support()]
    selected_df = pd.concat([y, X[selected_features]], axis=1)
    selected_df.to_csv(output_path, index=False)
    return selected_df, selected_features


if __name__ == "__main__":
    selected_df, selected_cols = select_features()
    print("Selected features:")
    print(selected_cols.tolist())
    print("\nSaved selected dataset as train_selected.csv")
    print(selected_df.head())
