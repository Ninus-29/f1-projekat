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

predictions = model.predict(X_test)

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.xlabel("Actual degradation")
plt.ylabel("Predicted degradation")
plt.title("Actual vs Predicted Degradation")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

print("Sacuvan grafik: actual_vs_predicted.png")