"""Join cutoff-safe operational features with specifications and known labels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("factoryiq.prepare")
ID_COLUMN = "vehicle_id"
TARGET_COLUMN = "target"


def prepare_training_data(
    features_path: Path = Path("data/train_features_100step.csv"),
    specifications_path: Path = Path("data/train_specifications.csv"),
    labels_path: Path = Path("data/train_labels_100step.csv"),
    output_path: Path = Path("data/modeling_dataset.csv"),
) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    specifications = pd.read_csv(specifications_path)
    labels = pd.read_csv(labels_path)

    for name, frame in (
        ("features", features),
        ("specifications", specifications),
        ("labels", labels),
    ):
        if ID_COLUMN not in frame:
            raise ValueError(f"{name} is missing {ID_COLUMN}")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} contains duplicate vehicle IDs")

    if TARGET_COLUMN not in labels:
        raise ValueError(f"Labels are missing {TARGET_COLUMN}")
    labels = labels.loc[labels[TARGET_COLUMN].notna()].copy()
    if not labels[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Known labels must be binary values 0 or 1")
    labels[TARGET_COLUMN] = labels[TARGET_COLUMN].astype("int8")

    joined = features.merge(
        specifications,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    ).merge(
        labels,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(joined) != len(labels):
        missing_features = labels.loc[~labels[ID_COLUMN].isin(joined[ID_COLUMN]), ID_COLUMN]
        raise ValueError(
            f"{len(missing_features)} labeled vehicles lack features or specifications"
        )
    if joined[ID_COLUMN].duplicated().any():
        raise RuntimeError("The modeling dataset contains duplicate vehicle IDs")
    forbidden = {"length_of_study_time_step", "in_study_repair"}
    if forbidden.intersection(joined.columns):
        raise ValueError("Repair outcome columns must not enter the modeling dataset")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    LOGGER.info(
        "Saved %s labeled vehicles to %s; target counts: %s",
        f"{len(joined):,}",
        output_path,
        joined[TARGET_COLUMN].value_counts().to_dict(),
    )
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/train_features_100step.csv"))
    parser.add_argument("--specifications", type=Path, default=Path("data/train_specifications.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels_100step.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/modeling_dataset.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_training_data(args.features, args.specifications, args.labels, args.output)


if __name__ == "__main__":
    main()