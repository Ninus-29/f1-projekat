import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("f1_tyre_degradation_dataset.csv")

y = df["degradation"]
X = df.drop(columns=["degradation"])

categorical_features = [
    "Driver",
    "Team",
    "Compound",
    "RaceName",
    "TrackStatus"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=30,
        random_state=42,
        n_jobs=2
    ))
])

print("Treniram model...")

model.fit(X_train, y_train)

print("Model istreniran")

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    model.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 atributa:")
print(importance_df.head(20))

importance_df.head(20).to_csv(
    "top20_features.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df.head(10)["Feature"],
    importance_df.head(10)["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Most Important Features")

plt.tight_layout()

plt.savefig("feature_importance.png")

print("\nSacuvan grafik: feature_importance.png")