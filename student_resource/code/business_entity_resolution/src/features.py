from rapidfuzz import fuzz
import pandas as pd
import numpy as np

def compute_pair_features(df1, df2, s1_to_s2, embs1, embs2):
    print("Computing pair features...")
    records = []
    
    # Convert DataFrames to lists/dicts for fast access
    s1_names = df1['norm_name'].fillna("").tolist()
    s2_names = df2['norm_name'].fillna("").tolist()
    
    s1_addrs = df1['norm_address'].fillna("").tolist()
    s2_addrs = df2['norm_address'].fillna("").tolist()
    
    s1_ids = df1['entity_id'].tolist()
    s2_ids = df2['entity_id'].tolist()
    
    # We create a dataframe of features
    for s1_idx, s2_indices in s1_to_s2.items():
        n1 = s1_names[s1_idx]
        a1 = s1_addrs[s1_idx]
        e1 = embs1[s1_idx]
        
        for s2_idx in s2_indices:
            n2 = s2_names[s2_idx]
            a2 = s2_addrs[s2_idx]
            e2 = embs2[s2_idx]
            
            # String similarities
            name_ratio = fuzz.ratio(n1, n2) / 100.0
            name_t_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
            name_jw = fuzz.partial_ratio(n1, n2) / 100.0
            
            addr_ratio = fuzz.ratio(a1, a2) / 100.0
            addr_t_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
            
            # JEPA similarity
            jepa_sim = np.dot(e1, e2)
            
            records.append({
                's1_idx': s1_idx,
                's2_idx': s2_idx,
                's1_id': s1_ids[s1_idx],
                's2_id': s2_ids[s2_idx],
                'name_ratio': name_ratio,
                'name_token_sort': name_t_sort,
                'name_jw': name_jw,
                'addr_ratio': addr_ratio,
                'addr_token_sort': addr_t_sort,
                'jepa_sim': jepa_sim
            })
            
    return pd.DataFrame(records)
