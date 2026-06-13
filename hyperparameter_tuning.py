import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("f1_tyre_degradation_dataset.csv")

print("Broj redova:", len(df))

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

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=2))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=2,
    verbose=2
)

print("\nPokrecem GridSearchCV...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nNajbolji hiperparametri:")
print(grid_search.best_params_)

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nRezultati najboljeg modela:")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))

tuning_results = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "Best Parameters": str(grid_search.best_params_),
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
}])

tuning_results.to_csv("hyperparameter_tuning_results.csv", index=False)

print("\nSacuvan fajl: hyperparameter_tuning_results.csv")