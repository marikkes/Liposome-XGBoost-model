import pandas as pd

def normalize_api_name(name):

    if pd.isna(name):
        return name

    return (
        str(name)
        .replace("-", "-")   # non-breaking hyphen U+2011
        .replace("–", "-")   # en dash U+2013
        .replace("—", "-")   # em dash U+2014
        .replace("\xa0", " ") # non-breaking space
        .replace("\u2011", "-")
        .strip()
        .replace("  ", " ")
    )