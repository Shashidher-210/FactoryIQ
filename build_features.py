"""Build vehicle-level operational features using readings through step 100."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("factoryiq.features")
ID_COLUMN = "vehicle_id"
TIME_COLUMN = "time_step"
DEFAULT_INPUT = Path("data/train_operational_readouts.csv")
DEFAULT_OUTPUT = Path("data/train_features_100step.csv")
DEFAULT_SPECIFICATIONS = Path("data/train_specifications.csv")


def build_features(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    specifications_path: Path = DEFAULT_SPECIFICATIONS,
    cutoff: int = 100,
    chunksize: int = 20_000,
) -> pd.DataFrame:
    """Aggregate operational readings without loading the complete CSV."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    specs = pd.read_csv(specifications_path, usecols=[ID_COLUMN])
    if specs[ID_COLUMN].duplicated().any():
        raise ValueError("Vehicle specifications contain duplicate vehicle IDs")
    vehicle_ids = specs[ID_COLUMN].to_numpy()
    vehicle_positions = {vehicle_id: position for position, vehicle_id in enumerate(vehicle_ids)}

    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    if ID_COLUMN not in header or TIME_COLUMN not in header:
        raise ValueError("Operational data must include vehicle_id and time_step")
    feature_columns = [column for column in header if column not in {ID_COLUMN, TIME_COLUMN}]
    if not feature_columns:
        raise ValueError("Operational data contains no measurement columns")

    vehicle_count = len(vehicle_ids)
    feature_count = len(feature_columns)
    totals = np.zeros((vehicle_count, feature_count), dtype=np.float64)
    sum_squares = np.zeros_like(totals)
    valid_counts = np.zeros((vehicle_count, feature_count), dtype=np.int64)
    minima = np.full_like(totals, np.inf)
    maxima = np.full_like(totals, -np.inf)
    latest_values = np.full_like(totals, np.nan)
    latest_times = np.full(vehicle_count, -np.inf, dtype=np.float64)
    latest_orders = np.full(vehicle_count, -1, dtype=np.int64)
    reading_counts = np.zeros(vehicle_count, dtype=np.int64)
    seen_vehicles = np.zeros(vehicle_count, dtype=bool)

    total_rows = 0
    retained_rows = 0
    unknown_ids: set[object] = set()

    reader = pd.read_csv(
        input_path,
        usecols=[ID_COLUMN, TIME_COLUMN, *feature_columns],
        chunksize=chunksize,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        row_orders = np.arange(total_rows, total_rows + len(chunk), dtype=np.int64)
        total_rows += len(chunk)
        chunk[TIME_COLUMN] = pd.to_numeric(chunk[TIME_COLUMN], errors="coerce")
        early = chunk.loc[chunk[TIME_COLUMN].le(cutoff)].copy()
        if early.empty:
            continue

        positions = early[ID_COLUMN].map(vehicle_positions)
        if positions.isna().any():
            unknown_ids.update(early.loc[positions.isna(), ID_COLUMN].unique().tolist())
            early = early.loc[positions.notna()].copy()
            positions = positions.loc[positions.notna()]
        if early.empty:
            continue

        row_order_by_index = pd.Series(row_orders, index=chunk.index)
        early["_source_order"] = row_order_by_index.loc[early.index].to_numpy()
        numeric = early[feature_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.where(np.isfinite(numeric), np.nan)
        early.loc[:, feature_columns] = numeric

        grouped = early.groupby(ID_COLUMN, sort=False, observed=True)[feature_columns]
        grouped_ids = grouped.size().index
        grouped_positions = np.asarray([vehicle_positions[value] for value in grouped_ids], dtype=np.int64)
        chunk_counts = grouped.count().to_numpy(dtype=np.int64)
        chunk_sums = grouped.sum().to_numpy(dtype=np.float64)
        chunk_squares = (early[feature_columns].pow(2).groupby(early[ID_COLUMN], sort=False).sum()).to_numpy(dtype=np.float64)
        chunk_minima = grouped.min().to_numpy(dtype=np.float64)
        chunk_maxima = grouped.max().to_numpy(dtype=np.float64)

        totals[grouped_positions] += chunk_sums
        sum_squares[grouped_positions] += chunk_squares
        valid_counts[grouped_positions] += chunk_counts
        minima[grouped_positions] = np.fmin(minima[grouped_positions], chunk_minima)
        maxima[grouped_positions] = np.fmax(maxima[grouped_positions], chunk_maxima)
        reading_counts[grouped_positions] += grouped.size().to_numpy(dtype=np.int64)
        seen_vehicles[grouped_positions] = True

        last_rows = (
            early.sort_values([ID_COLUMN, TIME_COLUMN, "_source_order"], kind="mergesort")
            .drop_duplicates(ID_COLUMN, keep="last")
        )
        last_positions = last_rows[ID_COLUMN].map(vehicle_positions).to_numpy(dtype=np.int64)
        last_times = last_rows[TIME_COLUMN].to_numpy(dtype=np.float64)
        last_orders = last_rows["_source_order"].to_numpy(dtype=np.int64)
        is_newer = (last_times > latest_times[last_positions]) | (
            (last_times == latest_times[last_positions])
            & (last_orders > latest_orders[last_positions])
        )
        selected_positions = last_positions[is_newer]
        latest_times[selected_positions] = last_times[is_newer]
        latest_orders[selected_positions] = last_orders[is_newer]
        latest_values[selected_positions] = last_rows.loc[is_newer, feature_columns].to_numpy(dtype=np.float64)

        retained_rows += len(early)
        if chunk_number % 10 == 0:
            LOGGER.info(
                "Processed %s rows; retained %s readings through step %s",
                f"{total_rows:,}",
                f"{retained_rows:,}",
                cutoff,
            )

    if unknown_ids:
        LOGGER.warning("Ignored %s operational IDs absent from specifications", len(unknown_ids))

    selected = np.flatnonzero(seen_vehicles)
    counts = valid_counts[selected]
    means = np.divide(totals[selected], counts, out=np.full_like(totals[selected], np.nan), where=counts > 0)
    numerator = sum_squares[selected] - np.divide(
        totals[selected] ** 2,
        counts,
        out=np.zeros_like(totals[selected]),
        where=counts > 0,
    )
    variances = np.divide(
        np.maximum(numerator, 0),
        counts - 1,
        out=np.full_like(numerator, np.nan),
        where=counts > 1,
    )
    standard_deviations = np.sqrt(variances)
    selected_minima = minima[selected].copy()
    selected_maxima = maxima[selected].copy()
    selected_minima[counts == 0] = np.nan
    selected_maxima[counts == 0] = np.nan
    feature_frames = {
        "mean": means,
        "std": standard_deviations,
        "min": selected_minima,
        "max": selected_maxima,
        "latest": latest_values[selected],
        "valid_count": counts.astype(np.float64),
        "missing_fraction": 1 - np.divide(
            counts,
            reading_counts[selected, None],
            out=np.zeros_like(counts, dtype=np.float64),
            where=reading_counts[selected, None] > 0,
        ),
    }
    data: dict[str, object] = {ID_COLUMN: vehicle_ids[selected]}
    for statistic, values in feature_frames.items():
        for column_index, column in enumerate(feature_columns):
            data[f"{column}__{statistic}"] = values[:, column_index]
    data["reading_count"] = reading_counts[selected]
    data["latest_time_step"] = latest_times[selected].astype(np.int64)
    result = pd.DataFrame(data)

    if result[ID_COLUMN].duplicated().any():
        raise RuntimeError("Feature aggregation produced duplicate vehicle IDs")
    if (result["latest_time_step"] > cutoff).any():
        raise RuntimeError("Feature aggregation included a reading beyond the cutoff")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    LOGGER.info(
        "Saved %s vehicle rows and %s features to %s (%s input rows, %s retained)",
        f"{len(result):,}",
        f"{result.shape[1] - 1:,}",
        output_path,
        f"{total_rows:,}",
        f"{retained_rows:,}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--specifications", type=Path, default=DEFAULT_SPECIFICATIONS)
    parser.add_argument("--cutoff", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=20_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_features(args.input, args.output, args.specifications, args.cutoff, args.chunksize)


if __name__ == "__main__":
    main()