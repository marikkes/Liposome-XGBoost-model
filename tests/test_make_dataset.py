import sqlite3

import pandas as pd
import pytest

from make_dataset import make_dataset
from get_api_profile import get_api_profile, preprocess_api_profile


def _write_table(db_path, table_name, frame):
    with sqlite3.connect(db_path) as conn:
        frame.to_sql(table_name, conn, index=False, if_exists="replace")


def test_make_dataset_builds_expected_features(tmp_path):
    db_path = tmp_path / "formulations_work.db"
    api_db_path = tmp_path / "api_properties.db"
    lipid_db_path = tmp_path / "lipid_properties.db"

    formulations = pd.DataFrame(
        [
            {
                "api": "API-1",
                "lipid_1": "DPPC",
                "lipid_2": None,
                "lipid_3": None,
                "lipid_1_qty": 80.0,
                "lipid_2_qty": None,
                "lipid_3_qty": None,
                "api_qty": 20.0,
                "ee_mean": 0.75,
            }
        ]
    )
    api_properties = pd.DataFrame(
        [
            {
                "api": "API-1",
                "molecular_weight": 100.0,
                "logp": 2.5,
                "rotatable_bond_count": 3,
                "hbond_acceptor_count": 4,
                "hbond_donor_count": 1,
                "heavy_atom_count": 12,
                "tpsa": 25.0,
                "complexity": 300.0,
                "defined_atom_stereocenter_count": 0,
                "defined_bond_stereocenter_count": 0,
                "number_of_rings": 1,
            }
        ]
    )
    lipid_properties = pd.DataFrame(
        [
            {
                "lipid_name": "DPPC",
                "logp": 1.5,
                "hbond_donor_count": 0,
                "tail_length_mean": 16.0,
            }
        ]
    )

    _write_table(db_path, "formulations", formulations)
    _write_table(api_db_path, "api_properties", api_properties)
    _write_table(lipid_db_path, "lipid_properties", lipid_properties)

    X, y, groups = make_dataset(db_path, api_db_path, lipid_db_path)
    row = X.iloc[0]

    assert len(X) == 1
    assert len(y) == 1
    assert list(groups) == ["API-1"]

    assert row["n_lipids"] == 1
    assert row["api_to_lipid_ratio"] == pytest.approx(0.25)
    assert row["lipid_DPPC_fraction"] == pytest.approx(1.0)
    assert row["lipid_1_fraction"] == pytest.approx(1.0)
    assert row["lipid_2_fraction"] == pytest.approx(0.0)
    assert row["lipid_3_fraction"] == pytest.approx(0.0)
    assert row["api_molecular_weight"] == pytest.approx(100.0)
    assert row["api_molecular_weight_missing"] == 0
    assert row["lipid_1_logp"] == pytest.approx(1.5)
    assert row["lipid_weighted_logp"] == pytest.approx(1.5)
    assert row["api_number_of_rings"] == 1
    assert row["api_number_of_rings_missing"] == 0
    assert y.iloc[0] == pytest.approx(0.75)


def test_make_dataset_imputes_missing_api_properties(tmp_path):
    db_path = tmp_path / "formulations_work.db"
    api_db_path = tmp_path / "api_properties.db"
    lipid_db_path = tmp_path / "lipid_properties.db"

    formulations = pd.DataFrame(
        [
            {
                "api": "API-1",
                "lipid_1": "DPPC",
                "lipid_2": None,
                "lipid_3": None,
                "lipid_1_qty": 50.0,
                "lipid_2_qty": None,
                "lipid_3_qty": None,
                "api_qty": 50.0,
                "ee_mean": 0.5,
            },
            {
                "api": "API-2",
                "lipid_1": "DPPC",
                "lipid_2": None,
                "lipid_3": None,
                "lipid_1_qty": 60.0,
                "lipid_2_qty": None,
                "lipid_3_qty": None,
                "api_qty": 40.0,
                "ee_mean": 0.6,
            },
        ]
    )
    api_properties = pd.DataFrame(
        [
            {
                "api": "API-1",
                "molecular_weight": 100.0,
                "logp": 2.0,
                "rotatable_bond_count": 3,
                "hbond_acceptor_count": 4,
                "hbond_donor_count": 1,
                "heavy_atom_count": 12,
                "tpsa": 25.0,
                "complexity": 300.0,
                "defined_atom_stereocenter_count": 0,
                "defined_bond_stereocenter_count": 0,
                "number_of_rings": 1,
            },
            {
                "api": "API-2",
                "molecular_weight": None,
                "logp": None,
                "rotatable_bond_count": 5,
                "hbond_acceptor_count": 2,
                "hbond_donor_count": None,
                "heavy_atom_count": 16,
                "tpsa": 31.0,
                "complexity": 450.0,
                "defined_atom_stereocenter_count": 0,
                "defined_bond_stereocenter_count": 1,
                "number_of_rings": 2,
            },
        ]
    )
    lipid_properties = pd.DataFrame(
        [
            {
                "lipid_name": "DPPC",
                "logp": 1.5,
                "hbond_donor_count": 0,
                "tail_length_mean": 16.0,
            }
        ]
    )

    _write_table(db_path, "formulations", formulations)
    _write_table(api_db_path, "api_properties", api_properties)
    _write_table(lipid_db_path, "lipid_properties", lipid_properties)

    X, y, groups = make_dataset(db_path, api_db_path, lipid_db_path)
    row1 = X.iloc[0]
    row2 = X.iloc[1]

    assert list(groups) == ["API-1", "API-2"]
    assert y.tolist() == pytest.approx([0.5, 0.6])


    assert row1["api_molecular_weight_missing"] == 0
    assert row2["api_molecular_weight_missing"] == 1
    assert row1["api_hbond_donor_count_missing"] == 0
    assert row2["api_hbond_donor_count_missing"] == 1
    assert row2["api_molecular_weight"] == pytest.approx(100.0)
    assert row2["api_logp"] == pytest.approx(2.0)

def test_api_profile_matches_dataset_features(tmp_path):
    # Create minimal dataset
    db_path = tmp_path / "formulations_work.db"
    api_db_path = tmp_path / "api_properties.db"
    lipid_db_path = tmp_path / "lipid_properties.db"

    formulations = pd.DataFrame([
        {
            "api": "API-1",
            "lipid_1": "DPPC",
            "lipid_2": None,
            "lipid_3": None,
            "lipid_1_qty": 100,
            "lipid_2_qty": None,
            "lipid_3_qty": None,
            "api_qty": 10,
            "ee_mean": 0.5,
        }
    ])

    api_properties = pd.DataFrame([
        {
            "api": "API-1",
            "molecular_weight": 100,
            "logp": 2,
            "rotatable_bond_count": 1,
            "hbond_acceptor_count": 2,
            "hbond_donor_count": 1,
            "heavy_atom_count": 10,
            "tpsa": 20,
            "complexity": 100,
            "defined_atom_stereocenter_count": 0,
            "defined_bond_stereocenter_count": 0,
            "number_of_rings": 2,
        }
    ])

    lipid_properties = pd.DataFrame([
        {
            "lipid_name": "DPPC",
            "logp": 1.5,
            "hbond_donor_count": 0,
            "tail_length_mean": 16,
        }
    ])

    _write_table(db_path, "formulations", formulations)
    _write_table(api_db_path, "api_properties", api_properties)
    _write_table(lipid_db_path, "lipid_properties", lipid_properties)

    X, _, _ = make_dataset(
        db_path,
        api_db_path,
        lipid_db_path
    )

    api_profile = get_api_profile(
        "API-1",
        api_properties
    )

    api_profile = preprocess_api_profile(
        api_profile,
        X
    )

    # Every API feature required by the model exists
    assert set(api_profile.keys()).issubset(
        set(X.columns)
    )

    # No missing values remain
    assert not any(
        pd.isna(v)
        for v in api_profile.values()
    )