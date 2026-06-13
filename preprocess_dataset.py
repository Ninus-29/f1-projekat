import pandas as pd
import numpy as np

# 1. Ucitavanje raw dataseta
df = pd.read_csv("f1_laps_2023_2025_raw.csv", low_memory=False)

print("Pocetni broj redova:", len(df))

# 2. Konverzija LapTime u sekunde
df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds()

# 3. Uklanjanje krugova bez osnovnih podataka
df = df.dropna(subset=["LapTimeSeconds", "Driver", "Compound", "TyreLife", "Stint"])

# 4. Uklanjanje pit in i pit out krugova
if "PitInTime" in df.columns:
    df = df[df["PitInTime"].isna()]

if "PitOutTime" in df.columns:
    df = df[df["PitOutTime"].isna()]

# 5. Sortiranje podataka
df = df.sort_values(["Season", "RaceName", "Driver", "Stint", "LapNumber"])

# 6. Racunanje prvog vremena u svakom stintu
df["FirstLapStintTime"] = df.groupby(
    ["Season", "RaceName", "Driver", "Stint"]
)["LapTimeSeconds"].transform("first")

# 7. Ciljna promenljiva - degradacija guma
df["degradation"] = df["LapTimeSeconds"] - df["FirstLapStintTime"]

# 8. Dodatni atributi
df["average_last_3_laps"] = df.groupby(
    ["Season", "RaceName", "Driver", "Stint"]
)["LapTimeSeconds"].transform(lambda x: x.shift(1).rolling(3).mean())

df["lap_time_delta"] = df.groupby(
    ["Season", "RaceName", "Driver", "Stint"]
)["LapTimeSeconds"].diff()

df["stint_length"] = df.groupby(
    ["Season", "RaceName", "Driver", "Stint"]
)["LapNumber"].transform("count")

df["tyre_age_ratio"] = df["TyreLife"] / df["stint_length"]

# 9. Popunjavanje weather kolona ako postoje
weather_columns = [
    "AirTemp", "Humidity", "Pressure", "Rainfall",
    "TrackTemp", "WindDirection", "WindSpeed"
]

for col in weather_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        df[col] = df.groupby(
            ["Season", "RaceName"]
        )[col].transform(lambda x: x.ffill().bfill())

        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = 0

# 10. Ako TrackStatus postoji, cuvamo ga; ako ne postoji, pravimo kolonu
if "TrackStatus" not in df.columns:
    df["TrackStatus"] = "1"

# 11. Izbor kolona za finalni dataset
columns = [
    "Season", "RaceName", "Driver", "Team",
    "LapNumber", "Stint", "Compound", "TyreLife",
    "Position", "TrackStatus",
    "AirTemp", "Humidity", "Pressure",
    "TrackTemp", "WindDirection", "WindSpeed",
    "LapTimeSeconds",
    "average_last_3_laps",
    "lap_time_delta",
    "tyre_age_ratio",
    "degradation"
]

df_final = df[columns].copy()

# 12. Uklanjamo samo redove gde fale najbitnije ML vrednosti
required_columns = [
    "LapTimeSeconds",
    "average_last_3_laps",
    "lap_time_delta",
    "tyre_age_ratio",
    "degradation"
]

df_final = df_final.dropna(subset=required_columns)

# 13. Uklanjamo ekstremno cudne vrednosti degradacije
df_final = df_final[(df_final["degradation"] > -10) & (df_final["degradation"] < 30)]

# 14. Cuvanje finalnog dataseta
df_final.to_csv("f1_tyre_degradation_dataset.csv", index=False)

print("Finalni broj redova:", len(df_final))
print("Sacuvan fajl: f1_tyre_degradation_dataset.csv")
print(df_final.head())