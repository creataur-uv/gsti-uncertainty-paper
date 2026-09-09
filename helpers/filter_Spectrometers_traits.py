## Data Cleaning

import pandas

def filter(in_Data, spectrometers, traits):
    refl_cols = in_Data.filter(regex=r"^Wave_ ").columns
    specs = list(spectrometers)
    traits = list(traits)
    df = in_Data[in_Data["Spectrometer"].isin(specs)]
    clean_wavelengths = df.dropna(subset=refl_cols)
    clean_Data =clean_wavelengths.dropna(subset=traits)
    traits_cols = in_Data.filter(traits).columns
    final_cols = traits_cols.append(refl_cols)
    return clean_wavelengths[final_cols].dropna()