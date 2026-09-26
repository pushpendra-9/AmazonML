import os
import pandas as pd
from data import load_data, preprocess_df
from jepa import train_jepa, get_jepa_embeddings
from blocking import get_tfidf_candidates, get_jepa_candidates, merge_candidates
from features import compute_pair_features
from matcher import train_matcher, predict_matches

def generate_output(s1_df, s23_df, cand_dict, matcher_model, embs1, embs2, output_dir, prefix=""):
    print(f"Generating features for {prefix} candidates...")
    features_df = compute_pair_features(s1_df, s23_df, cand_dict, embs1, embs2)
    
    print(f"Predicting matches for {prefix} set...")
    matches_df = predict_matches(matcher_model, features_df, threshold=0.5)
    
    cand_file = os.path.join(output_dir, f"{prefix}candidate_pairs.tsv")
    print(f"Writing {cand_file}...")
    with open(cand_file, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_idx, s23_indices in cand_dict.items():
            s1_id = s1_df.iloc[s1_idx]['entity_id']
            cand_ids = [s23_df.iloc[idx]['entity_id'] for idx in s23_indices]
            cand_ids = list(dict.fromkeys(cand_ids))
            f.write(f"{s1_id}\t{','.join(cand_ids)}\n")
            
        handled_s1 = set(s1_df.iloc[list(cand_dict.keys())]['entity_id'] if cand_dict else [])
        for s1_id in s1_df['entity_id']:
            if s1_id not in handled_s1:
                f.write(f"{s1_id}\t\n")
    
    match_file = os.path.join(output_dir, f"{prefix}matching_results.tsv")
    print(f"Writing {match_file}...")
    
    match_grouped = matches_df.groupby('s1_id')['s2_id'].apply(lambda x: list(dict.fromkeys(x))).to_dict()
    
    with open(match_file, "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_df['entity_id']:
            m_ids = match_grouped.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(m_ids)}\n")

def main():
    base_dir = "../../../" 
    train_dir = os.path.join(base_dir, "dataset", "train")
    test_dir = os.path.join(base_dir, "dataset", "test")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD & PREPROCESS FULL TRAINING DATA
    # ---------------------------------------------------------
    print("Loading Full Train Data...")
    s1_train = load_data(os.path.join(train_dir, "train_source1.tsv"))
    s2_train = load_data(os.path.join(train_dir, "train_source2.tsv"))
    s3_train = load_data(os.path.join(train_dir, "train_source3.tsv"))
    ground_truth = load_data(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    print("Preprocessing Train Data...")
    s1_train = preprocess_df(s1_train)
    s2_train = preprocess_df(s2_train)
    s3_train = preprocess_df(s3_train)
    s23_train = pd.concat([s2_train, s3_train]).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 2. JEPA REPRESENTATION LEARNING (UNLEASHED)
    # ---------------------------------------------------------
    all_texts = pd.concat([s1_train['combined_text'], s23_train['combined_text']]).dropna().tolist()
    jepa_encoder, jepa_tokenizer = train_jepa(all_texts, epochs=10, batch_size=1024)
    
    # ---------------------------------------------------------
    # 3. BLOCKING (CANDIDATE GENERATION) ON FULL TRAIN SET
    # ---------------------------------------------------------
    print("Extracting JEPA embeddings for Full Train Set...")
    s1_embs = get_jepa_embeddings(jepa_encoder, jepa_tokenizer, s1_train['combined_text'].fillna("").tolist())
    s23_embs = get_jepa_embeddings(jepa_encoder, jepa_tokenizer, s23_train['combined_text'].fillna("").tolist())
    
    print("Generating Candidates for Train Set...")
    cand_tfidf = get_tfidf_candidates(s1_train, s23_train, top_k=10)
    cand_jepa = get_jepa_candidates(s1_embs, s23_embs, top_k=10)
    s1_to_s23 = merge_candidates(cand_tfidf, cand_jepa)
    
    # ---------------------------------------------------------
    # 4. FEATURE ENGINEERING & LABELS FOR TRAIN SET
    # ---------------------------------------------------------
    print("Computing Train Features...")
    features_df = compute_pair_features(s1_train, s23_train, s1_to_s23, s1_embs, s23_embs)
    
    print("Preparing Labels...")
    gt_map = {}
    for _, row in ground_truth.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
        gt_map[s1_id] = set(m.strip() for m in matches if m.strip())
        
    def get_label(row):
        return 1 if row['s2_id'] in gt_map.get(row['s1_id'], set()) else 0
        
    features_df['label'] = features_df.apply(get_label, axis=1)
    
    # ---------------------------------------------------------
    # 5. TRAIN LIGHTGBM MATCHER
    # ---------------------------------------------------------
    print("Training LightGBM Matcher...")
    matcher_model = train_matcher(features_df, features_df['label'])
    
    # Free up memory before Test phase
    del s1_train, s2_train, s3_train, s23_train, s1_embs, s23_embs, features_df, cand_tfidf, cand_jepa, s1_to_s23
    import gc
    gc.collect()

    # ---------------------------------------------------------
    # 6. LOAD & PREPROCESS FULL TEST DATA
    # ---------------------------------------------------------
    print("\n--- Running Final Inference on Blind Test Set ---")
    s1_test = load_data(os.path.join(test_dir, "test_source1.tsv"))
    s2_test = load_data(os.path.join(test_dir, "test_source2.tsv"))
    s3_test = load_data(os.path.join(test_dir, "test_source3.tsv"))
    
    s1_test = preprocess_df(s1_test)
    s2_test = preprocess_df(s2_test)
    s3_test = preprocess_df(s3_test)
    s23_test = pd.concat([s2_test, s3_test]).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 7. INFERENCE PIPELINE
    # ---------------------------------------------------------
    print("Extracting JEPA embeddings for Test Split...")
    s1_test_embs = get_jepa_embeddings(jepa_encoder, jepa_tokenizer, s1_test['combined_text'].fillna("").tolist())
    s23_test_embs = get_jepa_embeddings(jepa_encoder, jepa_tokenizer, s23_test['combined_text'].fillna("").tolist())
    
    print("Generating Candidates for Test Set...")
    cand_tfidf_test = get_tfidf_candidates(s1_test, s23_test, top_k=10)
    cand_jepa_test = get_jepa_candidates(s1_test_embs, s23_test_embs, top_k=10)
    s1_to_s23_test = merge_candidates(cand_tfidf_test, cand_jepa_test)
    
    generate_output(s1_test, s23_test, s1_to_s23_test, matcher_model, s1_test_embs, s23_test_embs, output_dir, prefix="")
    
    print("Done! Outputs generated in output directory.")

if __name__ == "__main__":
    main()
