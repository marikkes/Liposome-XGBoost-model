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

def sort_lipids(chosen_lipids):
    """
    Sort lipids according to biological role.
    """

    def priority(col):
        lipid_name = (
            col.replace("lipid_", "")
               .replace("_fraction", "")
        )

        return LIPID_PRIORITY.get(lipid_name, 999)

    return sorted(chosen_lipids, key=priority)

def get_lipid_type_fraction_columns(X_columns):
    return [
        col for col in X_columns
        if col.startswith("lipid_")
        and col.endswith("_fraction")
        and not col.startswith("lipid_1")
        and not col.startswith("lipid_2")
        and not col.startswith("lipid_3")
    ]

def build_formulation_row(X_columns, chosen_lipids, weights, api_ratio, api_profile):
    row = dict.fromkeys(X_columns, 0)

    chosen_lipids = sort_lipids(chosen_lipids)

    for lipid, w in zip(chosen_lipids, weights):
        row[lipid] = w

    row["n_lipids"] = len(chosen_lipids)
    row["api_to_lipid_ratio"] = api_ratio

    for key, value in api_profile.items():
        if key in row:
            row[key] = value

    return row