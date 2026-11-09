import numpy as np
import pandas as pd

from build_features import build_features


def test_chunk_boundaries_and_cutoff_are_respected(tmp_path):
    operations_path = tmp_path / "operations.csv"
    specifications_path = tmp_path / "specifications.csv"
    output_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "vehicle_id": [10, 10, 10, 11],
            "time_step": [2, 100, 101, 100],
            "sensor_0": [1.0, 3.0, 999.0, np.nan],
        }
    ).to_csv(operations_path, index=False)
    pd.DataFrame({"vehicle_id": [10, 11]}).to_csv(specifications_path, index=False)

    result = build_features(
        input_path=operations_path,
        output_path=output_path,
        specifications_path=specifications_path,
        cutoff=100,
        chunksize=2,
    ).set_index("vehicle_id")

    assert result.loc[10, "sensor_0__mean"] == 2.0
    assert result.loc[10, "sensor_0__latest"] == 3.0
    assert result.loc[10, "reading_count"] == 2
    assert result.loc[10, "latest_time_step"] == 100
    assert result.loc[11, "sensor_0__missing_fraction"] == 1.0
    assert pd.isna(result.loc[11, "sensor_0__min"])
    assert not np.isinf(result.select_dtypes(include="number").to_numpy()).any()