import pandas as pd

def normalize_api_name(name):

    if pd.isna(name):
        return name

    return (
        str(name)
        .replace("\u2011", "-")   # non-breaking hyphen U+2011
        .replace("\u2013", "-")   # en dash U+2013
        .replace("\u2014", "-")   # em dash U+2014
        .replace("\xa0", " ") # non-breaking space
        .strip()
        .replace("  ", " ")
    )