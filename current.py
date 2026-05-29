import pandas as pd
p = "data/processed/dataset_v001"
labels = pd.read_parquet(f"{p}/labels_spectrum.parquet")
meta = pd.read_parquet(f"{p}/meta.parquet")
print(labels.to_csv("labels_spectrum.csv"))
print(meta.to_csv("meta.csv"))
