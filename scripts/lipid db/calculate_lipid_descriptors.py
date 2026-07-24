from pathlib import Path
import sqlite3
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    Lipinski,
    Descriptors,
    Crippen,
    rdMolDescriptors,
)
from rdkit.Chem.rdchem import BondStereo


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "lipid_properties.db"
)

def calculate_descriptors(smiles):

    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    
    # -----------------------
    # Defined stereocenters
    # -----------------------

    atom_stereocenters = len(
        Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=False
        )
    )

    bond_stereocenters = sum(
        bond.GetStereo() != BondStereo.STEREONONE
        for bond in mol.GetBonds()
    )

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "rotatable_bond_count": Lipinski.NumRotatableBonds(mol),
        "hbond_acceptor_count": Lipinski.NumHAcceptors(mol),
        "hbond_donor_count": Lipinski.NumHDonors(mol),
        "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
        "tpsa": Descriptors.TPSA(mol),
        "complexity": Descriptors.BertzCT(mol),
        "number_of_rings": Lipinski.RingCount(mol),
        "aromatic_ring_count": Lipinski.NumAromaticRings(mol),
        "aliphatic_ring_count": Lipinski.NumAliphaticRings(mol),
        "saturated_ring_count": Lipinski.NumSaturatedRings(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "heteroatom_count": Descriptors.NumHeteroatoms(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "atom_count": mol.GetNumAtoms(),
        "amide_bond_count": rdMolDescriptors.CalcNumAmideBonds(mol),
        "defined_atom_stereocenter_count": atom_stereocenters,
        "defined_bond_stereocenter_count": bond_stereocenters,
        "molar_refractivity": Crippen.MolMR(mol),
        "labute_surface_area": rdMolDescriptors.CalcLabuteASA(mol),
    }

def add_column_if_missing(conn, column_name):

    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA table_info(lipid_properties)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if column_name not in columns:

        cursor.execute(
            f"""
            ALTER TABLE lipid_properties
            ADD COLUMN {column_name} REAL
            """
        )

    conn.commit()



def calculate_lipid_descriptors():

    with sqlite3.connect(DB_PATH) as conn:

        df = pd.read_sql(
            "SELECT * FROM lipid_properties",
            conn
        )


        # -----------------------
        # Calculate descriptors
        # -----------------------

        descriptor_df = (
            df["smiles"]
            .apply(calculate_descriptors)
            .apply(pd.Series)
        )

        df.loc[
            descriptor_df.dropna(how="all").index,
            "descriptor_source"
        ] = "RDKit"

        # -----------------------
        # Add missing columns
        # -----------------------

        for column in descriptor_df.columns:
            add_column_if_missing(conn, column)

        # -----------------------
        # Merge descriptors
        # -----------------------

        df = pd.concat(
            [
                df,
                descriptor_df
            ],
            axis=1
        )

        # -----------------------
        # Update database
        # -----------------------

        calc_columns = descriptor_df.columns.tolist()

        calc_columns.append(
            "descriptor_source"
        )

        update_sql = f"""
            UPDATE lipid_properties
            SET {", ".join(f"{col} = ?" for col in calc_columns)}
            WHERE lipid_name = ?
        """

        for _, row in df.iterrows():

            values = [row[col] for col in calc_columns]
            values.append(row["lipid_name"])

            conn.execute(
                update_sql,
                values
            )


        conn.commit()