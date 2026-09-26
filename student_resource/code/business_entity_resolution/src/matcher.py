import lightgbm as lgb

def train_matcher(features_df, labels):
    print("Training LightGBM matcher...")
    feature_cols = ['name_ratio', 'name_token_sort', 'name_jw', 'addr_ratio', 'addr_token_sort', 'jepa_sim']
    X = features_df[feature_cols]
    y = labels
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        device_type='gpu'
    )
    model.fit(X, y)
    return model

def predict_matches(model, features_df, threshold=0.5):
    print("Predicting matches...")
    feature_cols = ['name_ratio', 'name_token_sort', 'name_jw', 'addr_ratio', 'addr_token_sort', 'jepa_sim']
    X = features_df[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    
    features_df['match_prob'] = probs
    features_df['predicted_match'] = probs >= threshold
    
    # Filter only predicted matches
    matches = features_df[features_df['predicted_match']]
    return matches
