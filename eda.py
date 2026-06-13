import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("f1_tyre_degradation_dataset.csv")

print("Broj redova:", len(df))
print("\nOpis numerickih atributa:")
print(df.describe())

# Korelacije
numeric_df = df.select_dtypes(include=["number"])

correlation_matrix = numeric_df.corr()

print("\nKorelacija sa degradation:")
corr = correlation_matrix["degradation"].sort_values(ascending=False)

print("\nTop 15 korelacija sa degradation:")
print(corr.head(15))

print("\nBottom 15 korelacija sa degradation:")
print(corr.tail(15))

# Histogram degradacije
plt.figure(figsize=(8,5))
plt.hist(df["degradation"], bins=50)
plt.title("Distribution of Tyre Degradation")
plt.xlabel("Degradation")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("degradation_distribution.png")

print("\nSacuvan grafik: degradation_distribution.png")

print("\nDegradation")
print("Min:", df["degradation"].min())
print("Max:", df["degradation"].max())

print("\nLapTimeSeconds")
print("Min:", df["LapTimeSeconds"].min())
print("Max:", df["LapTimeSeconds"].max())

print("\nTyreLife")
print("Min:", df["TyreLife"].min())
print("Max:", df["TyreLife"].max())

print("\nTyreLife statistika:")
print(df["TyreLife"].describe())

print("\nNajvece vrednosti TyreLife:")
print(
    df[["Driver", "RaceName", "TyreLife"]]
    .sort_values("TyreLife", ascending=False)
    .head(20)
)