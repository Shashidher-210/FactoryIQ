
import pandas as pd

# Load the SCANIA repair records
df = pd.read_csv("data/train_tte.csv")

# Display basic information
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nRepair distribution:")
print(df["in_study_repair"].value_counts())

print("\nRepair percentages:")
print(df["in_study_repair"].value_counts(normalize=True) * 100)

# Load vehicle specifications
specs = pd.read_csv("data/train_specifications.csv")

print("\nVehicle specifications shape:")
print(specs.shape)

print("\nVehicle specifications columns:")
print(specs.columns.tolist())

print("\nFirst 5 specification records:")
print(specs.head())

# Combine repair records and vehicle specifications
merged_df = pd.merge(
    df,
    specs,
    on="vehicle_id",
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset shape:")
print(merged_df.shape)

print("\nFirst 5 merged records:")
print(merged_df.head())

# Read only the first 5 operational records
operations = pd.read_csv(
    "data/train_operational_readouts.csv",
    nrows=5
)

print("\nOperational data shape (sample):")
print(operations.shape)

print("\nOperational data columns:")
print(operations.columns.tolist())

print("\nFirst 5 operational records:")
print(operations.iloc[:, :8])

# Read a sample of operational records
sample = pd.read_csv(
    "data/train_operational_readouts.csv",
    nrows=10000
)

print("\nUnique vehicles in first 10,000 readings:")
print(sample["vehicle_id"].nunique())

print("\nReadings per vehicle (first 10 vehicles in sample):")
print(sample.groupby("vehicle_id").size().head(10))

print("\nMissing values in first 10 columns:")
print(sample.iloc[:, :10].isnull().sum())

# Inspect operational data types
print("\nOperational data types:")
print(sample.dtypes.value_counts())

# Check the time range for vehicle 0
vehicle_0 = sample[sample["vehicle_id"] == 0]

print("\nVehicle 0 time range:")
print("Earliest:", vehicle_0["time_step"].min())
print("Latest:", vehicle_0["time_step"].max())

# Check whether time steps are ordered
print("\nVehicle 0 timestamps ordered:")
print(vehicle_0["time_step"].is_monotonic_increasing)
