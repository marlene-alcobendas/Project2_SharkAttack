""""
Normalize text columns (strip spaces, set lowercase).

Parameters
----------
df : DataFrame
cols : columns to normalize; if None, all object/string columns
lower : convert to lowercase
strip : strip leading/trailing whitespace
normalize_columns: normalize column names (strip, lower, replace spaces with _)

Returns
-------
DataFrame (same object, modified in place style but returns df for chaining)
"""
import pandas as pd
from typing import Optional, Iterable

def standardize_text(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    lower: bool = True,
    strip: bool = True,
    normalize_columns: bool = True,
) -> pd.DataFrame:
       
    if cols is None:
        cols = df.select_dtypes(include=["object", "string"]).columns

    for c in cols:
        s = df[c].astype("string")
        if strip:
            s = s.str.strip()
        if lower:
            s = s.str.lower()
        df[c] = s
     
    if normalize_columns:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    return df