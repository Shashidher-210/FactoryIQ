
import pandas as pd

# Read a small sample to verify our preprocessing setup
operations = pd.read_csv(
    "data/train_operational_readouts.csv",
    nrows=10000
)

# Sort readings by vehicle and time
operations = operations.sort_values(
    ["vehicle_id", "time_step"]
)

print("Operational sample loaded successfully!")
print("Shape:", operations.shape)
print("Unique vehicles:", operations["vehicle_id"].nunique())

# Identify operational feature columns
feature_columns = [
    col for col in operations.columns
    if col not in ["vehicle_id", "time_step"]
]

print("Number of operational features:", len(feature_columns))

# Select a few operational features for our first test
selected_features = ["171_0", "666_0", "427_0"]

# Calculate rolling averages using current and past readings
for feature in selected_features:
    operations[f"{feature}_rolling_avg"] = (
        operations.groupby("vehicle_id")[feature]
        .transform(
            lambda x: x.rolling(
                window=5,
                min_periods=1
            ).mean()
        )
    )

# Display the original and rolling values for vehicle 0
print("\nVehicle 0 historical features:")
print(
    operations.loc[
        operations["vehicle_id"] == 0,
        [
            "vehicle_id",
            "time_step",
            "171_0",
            "171_0_rolling_avg"
        ]
    ].head(10)
)

# Summarize the selected operational features per vehicle
vehicle_features = operations.groupby("vehicle_id").agg(
    reading_count=("time_step", "count"),
    first_reading_time=("time_step", "min"),
    last_reading_time=("time_step", "max"),
    avg_171_0=("171_0", "mean"),
    avg_666_0=("666_0", "mean"),
    avg_427_0=("427_0", "mean")
).reset_index()

print("\nVehicle-level feature summary:")
print(vehicle_features.head())

print("\nVehicle-level dataset shape:")
print(vehicle_features.shape)

# Load vehicle repair outcomes
repair_data = pd.read_csv("data/train_tte.csv")

# Join our exploratory features with repair outcomes
vehicle_summary = vehicle_features.merge(
    repair_data,
    on="vehicle_id",
    how="left",
    validate="one_to_one"
)

print("\nCombined vehicle summary shape:")
print(vehicle_summary.shape)

print("\nRepair outcomes in sampled vehicles:")
print(vehicle_summary["in_study_repair"].value_counts(dropna=False))

print("\nFirst 5 combined records:")
print(vehicle_summary.head())

# Define a common prediction time
prediction_time = 100

# Keep only measurements available by time step 100
early_operations = operations[
    operations["time_step"] <= prediction_time
].copy()

print("\nPrediction cutoff:", prediction_time)

print("\nReadings available before cutoff:")
print(len(early_operations))

print("\nVehicles with early readings:")
print(early_operations["vehicle_id"].nunique())

print("\nLatest available reading:")
print(early_operations["time_step"].max())

# Define a 48-step prediction window
prediction_horizon = 100
window_end = prediction_time + prediction_horizon

# Use only vehicles with readings available by prediction time
eligible_vehicles = repair_data[
    repair_data["vehicle_id"].isin(
        early_operations["vehicle_id"].unique()
    )
].copy()

event_time = eligible_vehicles["length_of_study_time_step"]
repaired = eligible_vehicles["in_study_repair"] == 1

# Repair observed within the future prediction window
positive = (
    repaired
    & (event_time > prediction_time)
    & (event_time <= window_end)
)

# No repair observed during the full prediction window
negative = (
    (event_time >= window_end)
    & ~positive
    & (~repaired | (event_time > window_end))
)

eligible_vehicles["target"] = pd.NA
eligible_vehicles.loc[positive, "target"] = 1
eligible_vehicles.loc[negative, "target"] = 0

print("\n100-step prediction target distribution:")
print(
    eligible_vehicles["target"]
    .value_counts(dropna=False)
)

print("\nUsable labeled vehicles:")
print(eligible_vehicles["target"].notna().sum())

# Examine repair times among sampled vehicles
repair_cases = eligible_vehicles[
    eligible_vehicles["in_study_repair"] == 1
].copy()

print("\nTotal repair cases in sample:")
print(len(repair_cases))

print("\nRepair time statistics:")
print(repair_cases["length_of_study_time_step"].describe())

print("\nRepair times in ascending order:")
print(
    repair_cases["length_of_study_time_step"]
    .sort_values()
    .to_list()
)
