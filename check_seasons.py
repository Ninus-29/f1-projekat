import pandas as pd

df = pd.read_csv("f1_tyre_degradation_dataset.csv")

print(df["Season"].value_counts())

print("\nBroj trka po sezoni:")
print(df.groupby("Season")["RaceName"].nunique())