import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "R2": [
        0.6102,
        0.8340,
        0.9422,
        0.9013
    ]
})

plt.figure(figsize=(8,5))
plt.bar(results["Model"], results["R2"])

plt.title("Model Comparison by R² Score")
plt.xlabel("Model")
plt.ylabel("R²")

plt.tight_layout()

plt.savefig("model_comparison.png")

print("Sacuvan grafik: model_comparison.png")