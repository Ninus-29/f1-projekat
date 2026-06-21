import streamlit as st
import pandas as pd
import joblib

model = joblib.load("f1_tyre_degradation_model.pkl")
df = pd.read_csv("f1_tyre_degradation_dataset.csv")

st.set_page_config(
    page_title="F1 Tyre Degradation Prediction",
    layout="centered"
)

st.title("🏎️ Formula 1 Tyre Degradation Prediction")

mode = st.radio(
    "Choose mode",
    ["Validate Existing Data", "Predict New Scenario"]
)

feature_columns = [
    "Season", "RaceName", "Driver", "Team",
    "LapNumber", "Stint", "Compound", "TyreLife",
    "Position", "TrackStatus",
    "AirTemp", "Humidity", "Pressure",
    "TrackTemp", "WindDirection", "WindSpeed",
    "LapTimeSeconds",
    "average_last_3_laps",
    "lap_time_delta",
    "tyre_age_ratio",
    "compound_factor",
    "compound_tyrelife"
    ]

if mode == "Validate Existing Data":

    st.write(
        "Izaberite postojeću trku, vozača, stint i krug iz dataseta. "
        "Aplikacija poredi predikciju modela sa stvarnom vrednošću."
    )

    race_name = st.selectbox(
        "Race",
        sorted(df["RaceName"].dropna().unique())
    )

    race_df = df[df["RaceName"] == race_name]

    driver = st.selectbox(
        "Driver",
        sorted(race_df["Driver"].dropna().unique())
    )

    driver_df = race_df[race_df["Driver"] == driver]

    stint = st.selectbox(
        "Stint",
        sorted(driver_df["Stint"].dropna().unique())
    )

    stint_df = driver_df[driver_df["Stint"] == stint]

    lap_number = st.selectbox(
        "Lap Number",
        sorted(stint_df["LapNumber"].dropna().unique())
    )

    selected_row = stint_df[stint_df["LapNumber"] == lap_number].iloc[0]

    st.subheader("Selected lap data")

    st.write({
        "Season": int(selected_row["Season"]),
        "Race": selected_row["RaceName"],
        "Driver": selected_row["Driver"],
        "Team": selected_row["Team"],
        "Compound": selected_row["Compound"],
        "Lap Number": int(selected_row["LapNumber"]),
        "Stint": int(selected_row["Stint"]),
        "Tyre Life": int(selected_row["TyreLife"]),
        "Position": int(selected_row["Position"]),
        "Lap Time Seconds": round(selected_row["LapTimeSeconds"], 3)
    })

    input_data = pd.DataFrame([selected_row[feature_columns]])

    if st.button("Predict Degradation"):
        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted difference from first lap of stint: {prediction:.3f} seconds"
        )

        st.info(
            f"Actual value in dataset: {selected_row['degradation']:.3f} seconds"
        )

else:

    st.write(
        "Unesite novu kombinaciju uslova. "
        "Za vrednosti koje korisnik ne unosi ručno koriste se prosečne vrednosti iz dataseta."
    )

    driver = st.selectbox(
        "Driver",
        sorted(df["Driver"].dropna().unique())
    )

    team = st.selectbox(
        "Team",
        sorted(df["Team"].dropna().unique())
    )

    race_name = st.selectbox(
        "Race",
        sorted(df["RaceName"].dropna().unique())
    )

    compound = st.selectbox(
        "Compound",
        sorted(df["Compound"].dropna().unique())
    )

    season = st.number_input(
        "Season",
        min_value=2023,
        max_value=2026,
        value=2026
    )

    lap_number = st.number_input(
        "Lap Number",
        min_value=1,
        max_value=100,
        value=20
    )

    stint = st.number_input(
        "Stint",
        min_value=1,
        max_value=10,
        value=2
    )

    tyre_life = st.number_input(
        "Tyre Life",
        min_value=1,
        max_value=100,
        value=15
    )

    position = st.number_input(
        "Position",
        min_value=1,
        max_value=20,
        value=5
    )

    track_status = str(df["TrackStatus"].mode()[0])

    air_temp = df["AirTemp"].mean()
    humidity = df["Humidity"].mean()
    pressure = df["Pressure"].mean()
    track_temp = df["TrackTemp"].mean()
    wind_direction = df["WindDirection"].mean()
    wind_speed = df["WindSpeed"].mean()

    base_lap_time = df[df["RaceName"] == race_name]["LapTimeSeconds"].mean()

    if pd.isna(base_lap_time):
        base_lap_time = df["LapTimeSeconds"].mean()

    expected_degradation = tyre_life * 0.08

    lap_time_seconds = base_lap_time + expected_degradation
    average_last_3_laps = lap_time_seconds - 0.1
    lap_time_delta = 0.08
    tyre_age_ratio = tyre_life / 30

    compound_factor_map = {
    "SOFT": 1.5,
    "MEDIUM": 1.0,
    "HARD": 0.6,
    "INTERMEDIATE": 1.2,
    "WET": 1.1
}

    compound_factor = compound_factor_map.get(compound, 1.0)
    compound_tyrelife = tyre_life * compound_factor

    input_data = pd.DataFrame([{
        "Season": season,
        "RaceName": race_name,
        "Driver": driver,
        "Team": team,
        "LapNumber": lap_number,
        "Stint": stint,
        "Compound": compound,
        "TyreLife": tyre_life,
        "Position": position,
        "TrackStatus": track_status,
        "AirTemp": air_temp,
        "Humidity": humidity,
        "Pressure": pressure,
        "TrackTemp": track_temp,
        "WindDirection": wind_direction,
        "WindSpeed": wind_speed,
        "LapTimeSeconds": lap_time_seconds,
        "average_last_3_laps": average_last_3_laps,
        "lap_time_delta": lap_time_delta,
        "tyre_age_ratio": tyre_age_ratio,
        "compound_factor": compound_factor,
        "compound_tyrelife": compound_tyrelife
    }])

    if st.button("Predict New Degradation"):
        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted difference from first lap of stint: {prediction:.3f} seconds"
        )

        st.caption(
            "Positive value means the lap is predicted to be slower than the first lap of the stint. "
            "Negative value means it is predicted to be faster than the first lap of the stint."
        )