import os
import fastf1
import pandas as pd

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

seasons = [2023, 2024, 2025]
all_laps = []

for year in seasons:
    print(f"\nUčitavam sezonu {year}...")

    schedule = fastf1.get_event_schedule(year)

    races = schedule[schedule["EventFormat"] != "testing"]

    for _, race in races.iterrows():
        race_name = race["EventName"]

        try:
            print(f"Učitavam: {year} - {race_name}")

            session = fastf1.get_session(year, race_name, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=False)

            laps = session.laps.copy()

            laps["Season"] = year
            laps["RaceName"] = race_name

            weather = session.weather_data.copy()

            laps = laps.merge(
                weather,
                on="Time",
                how="left"
            )

            all_laps.append(laps)

            print(f"Uspešno: {year} - {race_name}")

        except Exception as e:
            print(f"Greška za {year} - {race_name}: {e}")

dataset = pd.concat(all_laps, ignore_index=True)

dataset.to_csv("f1_laps_2023_2025_raw.csv", index=False)

print("\nGotovo!")
print(f"Ukupno redova: {len(dataset)}")
print("Sačuvan fajl: f1_laps_2023_2025_raw.csv")