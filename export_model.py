import pandas as pd
import joblib

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
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=2
    ))
])

print("Treniram finalni model...")
model.fit(X_train, y_train)

joblib.dump(model, "f1_tyre_degradation_model.pkl")

print("Model je eksportovan: f1_tyre_degradation_model.pkl")