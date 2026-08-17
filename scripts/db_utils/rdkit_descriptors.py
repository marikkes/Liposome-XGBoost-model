import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    Lipinski,
    Descriptors,
    Crippen,
    rdMolDescriptors,
)
from rdkit.Chem.rdchem import BondStereo


def calculate_rdkit_descriptors(smiles):

    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None


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