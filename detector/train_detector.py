import argparse
import os
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to labelled .pt records file")
    p.add_argument(
        "--model", required=True, help="Model name (used for output filenames)"
    )
    p.add_argument("--outdir", default=".", help="Directory to save plots")
    p.add_argument(
        "--label-key",
        default="hallucination_label",
        help="Key inside metadata dict that holds 0/1 hallucination label",
    )
    return p.parse_args()

def load_data(file, label_key):

    df = pd.read_pickle(file)
    print(f"Loaded {len(df)} records from {file}")
    print(df.head())
    if label_key not in df.columns:
        raise ValueError(f"Label key {label_key} not found in dataframe columns")
    return df, label_key


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, label_key = load_data(args.file, args.label_key)

if __name__ == "__main__":
    main()
