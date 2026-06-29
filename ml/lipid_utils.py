LIPID_PRIORITY = {
    # Main phospholipids
    "DPPC": 1,
    "DSPC": 1,
    "DMPC": 1,
    "DAPC": 1,
    "Egg PC": 1,
    "Soy PC": 1,

    # Helper lipids
    "DOPE": 2,
    "DPPG": 2,
    "CHEMS": 2,

    # Cationic lipids
    "DOTAP": 3,

    # Sterols LAST
    "Chol": 4,
}

# List of lipids to exclude from suggestions and optimization due to lab availability
EXCLUDED_LIPIDS = {
    "CHEMS",
    "DPPG",
}

def get_lipid_type_fraction_columns(X_columns):
    return [
        col for col in X_columns
        if col.startswith("lipid_")
        and col.endswith("_fraction")
        and not col.startswith("lipid_1")
        and not col.startswith("lipid_2")
        and not col.startswith("lipid_3")
    ]

def get_available_lipids(X_columns):
    lipids = get_lipid_type_fraction_columns(X_columns)

    available = []
    
    for col in lipids:
        lipid_name = lipid_name_from_column(col)
        
        if lipid_name not in EXCLUDED_LIPIDS:
            available.append(col)

    return available

def sort_lipids(chosen_lipids):
    """
    Sort lipids according to biological role.
    """

    def priority(col):
        lipid_name = lipid_name_from_column(col)
        return LIPID_PRIORITY.get(lipid_name, 999)

    return sorted(chosen_lipids, key=priority)

def extract_present_lipids(row, lipid_cols, threshold=0.01):
    """
    Returns dict: {lipid_column: value}
    only for lipids above threshold.
    """
    return {
        col: float(row[col])
        for col in lipid_cols
        if row[col] > threshold
    }

def lipid_name_from_column(col: str) -> str:
    """
    Convert dataframe column name -> clean lipid name.
    """
    return col.replace("lipid_", "").replace("_fraction", "")

def lipid_column_from_name(name: str) -> str:
    """
    Convert clean lipid name -> dataframe column name.
    (Useful for safe reconstruction if needed later)
    """
    return f"lipid_{name}_fraction"