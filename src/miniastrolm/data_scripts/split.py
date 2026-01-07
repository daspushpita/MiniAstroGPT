import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_jason, split_ratio, seed=42):
    """Splits the data into train and validation sets

    Args:
        input_jason (_type_): input jason file path
        split_ratio (_type_): ratio for train and validation split
        seed (int, optional): _description_. Defaults to 42.
    """
    df = pd.read_json(input_jason, lines=True, orient="records")
    train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=42)

    train_df.to_json("train_data.jsonl", lines=True, orient="records", force_ascii=False)
    val_df.to_json("val_data.jsonl", lines=True, orient="records", force_ascii=False)
    print(f"Train: {len(train_df)}, Val: {len(val_df)} saved to JSONL files")
    return train_df, val_df

data = split_data(input_jason="astro_abstracts_2025.jsonl", split_ratio=0.1)