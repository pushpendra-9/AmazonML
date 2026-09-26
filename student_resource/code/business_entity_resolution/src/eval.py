import pandas as pd

def calculate_f05_macro(ground_truth_df, predictions_df):
    """
    Computes the macro-averaged F_0.5 score.
    Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Calculated per Source 1 entity, then averaged across all Source 1 entities.
    """
    print("Calculating macro F0.5 score...")
    
    # Create dict mapping s1_id -> set of ground truth matches
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() != "" else []
        gt_map[s1] = set(m.strip() for m in matches if m.strip())
        
    # Create dict mapping s1_id -> set of predicted matches
    pred_map = {}
    for _, row in predictions_df.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() != "" else []
        pred_map[s1] = set(m.strip() for m in matches if m.strip())
        
    f05_scores = []
    
    # Only evaluate entities that were part of the validation split (present in predictions)
    for s1_id in pred_map.keys():
        true_matches = gt_map.get(s1_id, set())
        pred_matches = pred_map.get(s1_id, set())
        
        # Singleton logic
        if len(true_matches) == 0:
            if len(pred_matches) == 0:
                f05_scores.append(1.0) # Correctly identified singleton
            else:
                f05_scores.append(0.0) # Falsely predicted matches for singleton
            continue
            
        if len(pred_matches) == 0:
            f05_scores.append(0.0) # Missed all matches for non-singleton
            continue
            
        # Calculate precision and recall
        true_positives = len(true_matches.intersection(pred_matches))
        precision = true_positives / len(pred_matches)
        recall = true_positives / len(true_matches)
        
        if precision == 0 and recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
            
    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    print(f"Total S1 Entities Evaluated: {len(f05_scores)}")
    print(f"Macro F0.5 Score: {macro_f05:.4f}")
    return macro_f05

if __name__ == "__main__":
    import os
    
    # Path assumption based on running locally in the project root
    base_dir = "../../../"
    gt_file = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    pred_file = os.path.join(base_dir, "output", "val_matching_results.tsv")
    
    if os.path.exists(gt_file) and os.path.exists(pred_file):
        gt_df = pd.read_csv(gt_file, sep='\t')
        pred_df = pd.read_csv(pred_file, sep='\t')
        calculate_f05_macro(gt_df, pred_df)
    else:
        print(f"Could not find {gt_file} or {pred_file}")
