import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("f1_tyre_degradation_dataset.csv")

top_features = [
    "Stint",
    "lap_time_delta",
    "LapTimeSeconds",
    "Season",
    "Position"
]

X = df[top_features]
y = df["degradation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=2
)

print("Treniram model sa najbitnijim atributima...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

results = pd.DataFrame([
    {
        "Feature set": "All features",
        "MAE": 0.388801,
        "MSE": 1.012789,
        "RMSE": 1.006374,
        "R2": 0.942218
    },
    {
        "Feature set": "Top 5 features",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
])

results.to_csv("top_features_comparison.csv", index=False)

print("\nRezultati poredjenja:")
print(results)

print("\nSacuvan fajl: top_features_comparison.csv")