
import pandas as pd

file_path = "data/train_operational_readouts.csv"

total_rows = 0
early_rows = 0
vehicle_ids = set()

# Read the large CSV in manageable chunks
for chunk in pd.read_csv(
    file_path,
    usecols=["vehicle_id", "time_step"],
    chunksize=20000
):
    total_rows += len(chunk)

    # Keep only readings available at prediction time
    early_chunk = chunk[chunk["time_step"] <= 100]

    early_rows += len(early_chunk)
    vehicle_ids.update(early_chunk["vehicle_id"].unique())

print("\nFull dataset analysis completed!")
print("Total operational readings:", total_rows)
print("Readings at or before time 100:", early_rows)
print("Vehicles with early readings:", len(vehicle_ids))


# Load repair records
repair_data = pd.read_csv("data/train_tte.csv")

# Keep vehicles with operational readings by time step 100
eligible = repair_data[
    repair_data["vehicle_id"].isin(vehicle_ids)
].copy()

prediction_time = 100
window_end = 200

event_time = eligible["length_of_study_time_step"]
repaired = eligible["in_study_repair"] == 1

# Repair observed between time steps 100 and 200
positive = (
    repaired
    & (event_time > prediction_time)
    & (event_time <= window_end)
)

# No repair during the full prediction window
negative = (
    (event_time >= window_end)
    & (~repaired | (event_time > window_end))
)

eligible["target"] = pd.NA
eligible.loc[positive, "target"] = 1
eligible.loc[negative, "target"] = 0

print("\nFull dataset target distribution:")
print(eligible["target"].value_counts(dropna=False))

print("\nUsable labeled vehicles:")
print(eligible["target"].notna().sum())

# Keep only vehicles with known outcomes
labeled_vehicles = eligible.dropna(
    subset=["target"]
).copy()

# Convert target labels to integers
labeled_vehicles["target"] = (
    labeled_vehicles["target"].astype(int)
)

# Save labels separately from operational features
labeled_vehicles[
    ["vehicle_id", "target"]
].to_csv(
    "data/train_labels_100step.csv",
    index=False
)

print("\nTraining labels saved successfully!")
print("Shape:", labeled_vehicles.shape)
print("Positive examples:", labeled_vehicles["target"].sum())
