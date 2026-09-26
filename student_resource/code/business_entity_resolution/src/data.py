import pandas as pd
import re

def load_data(file_path):
    print(f"Loading {file_path}...")
    return pd.read_csv(file_path, sep='\t')

def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_business_name(name):
    if pd.isna(name): return ""
    name = normalize_text(name)
    suffixes = [' limited', ' ltd', ' pvt', ' private', ' inc', ' corporation', ' corp', ' llc', ' co']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip()

def preprocess_df(df):
    print("Preprocessing data...")
    df['norm_name'] = df['business_name'].apply(normalize_business_name)
    df['norm_address'] = df['business_address'].apply(normalize_text)
    df['combined_text'] = df['norm_name'] + " " + df['norm_address']
    return df
