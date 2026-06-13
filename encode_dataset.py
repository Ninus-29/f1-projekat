import pandas as pd

df = pd.read_csv("f1_tyre_degradation_dataset.csv")

print("Pre enkodiranja:", df.shape)

categorical_columns = [
    "Driver",
    "Team",
    "Compound",
    "RaceName",
    "TrackStatus"
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("Posle enkodiranja:", df_encoded.shape)

df_encoded.to_csv(
    "f1_tyre_degradation_encoded.csv",
    index=False
)

print("Sacuvan fajl: f1_tyre_degradation_encoded.csv")