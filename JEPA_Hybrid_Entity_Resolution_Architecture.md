# JEPA + Hybrid Business Entity Resolution Architecture

> Competition-grade implementation specification.

## Overview

Build a hybrid Entity Resolution system where **JEPA is used for representation learning and retrieval**, while a **gradient-boosted pair classifier** makes the final matching decisions.

### High-Level Pipeline

```text
                         RAW RECORDS
                              │
                  ┌───────────┴───────────┐
                  │                       │
             normalization          raw representations
                  │                       │
                  └───────────┬───────────┘
                              │
                       MULTI-BLOCKING
                              │
               ┌──────────────┼──────────────┐
               │              │              │
          lexical blocks   TF-IDF ANN     JEPA ANN
               │              │              │
               └──────────────┼──────────────┘
                              │
                       candidate union
                              │
                     pair feature engine
                              │
      ┌───────────────────────┼─────────────────────────┐
      │                       │                         │
 lexical similarity      address features       JEPA features
      │                       │                         │
      └───────────────────────┼─────────────────────────┘
                              │
                  LightGBM / XGBoost Matcher
                              │
                 probability calibration
                              │
                  singleton / abstention gate
                              │
                optional graph consistency
                              │
                      matching_results.tsv
```

## Core Components

### 1. Data Loading
- Read TSV files with `sep="\\t"`.
- Treat `country` as an open string field.
- Infer source from file or entity ID prefix.

### 2. Normalization
Maintain:
- raw text
- normalized text
- core business name (suffix removed)
- compact alphanumeric form

Normalize:
- Unicode
- punctuation
- whitespace
- `& ↔ and`
- common legal suffixes

### 3. Address Parsing
Extract heuristically:
- house number
- postal code
- city
- state
- district
- street tokens

Create component agreement features.

### 4. JEPA Representation Learning
Train using multiple corrupted views of each business record.

Corruptions:
- abbreviation changes
- punctuation removal
- token deletion
- typos
- address component dropout
- reordered address tokens

Use:
- shared encoder
- predictor
- EMA target encoder (optional)

Loss:

```text
L =
  λ1 * JEPA
+ λ2 * Contrastive
+ λ3 * Pair Classification
```

### 5. Cross-Source Learning

Positive pairs:
- S1 ↔ S2
- S1 ↔ S3

Train embeddings so matched entities are close while hard negatives are far apart.

### 6. Candidate Generation

Union of:
- exact normalized blocking
- rare-token blocking
- address blocking
- character TF-IDF retrieval
- JEPA ANN retrieval

Save this exact set to `candidate_pairs.tsv`.

### 7. Pair Features

Include:
- RapidFuzz similarities
- Jaro-Winkler
- Levenshtein
- TF-IDF cosine
- token overlap
- address component agreement
- country agreement
- JEPA cosine similarities
- cross-feature interactions

### 8. Matcher

Train separate models:
- S1 ↔ S2
- S1 ↔ S3

Recommended:
- LightGBM
- XGBoost (optional ensemble)

Train only on blocked candidates.

### 9. Hard Negative Mining

Mine negatives from:
- high lexical similarity
- high TF-IDF similarity
- high embedding similarity
- same city/postal but different business

Avoid relying on random negatives.

### 10. Calibration

Use:
- Platt Scaling or
- Isotonic Regression

Optimize thresholds directly for **macro F0.5**.

### 11. Singleton Gate

Estimate whether an S1 entity has any match.

Suppress weak predictions to improve precision.

### 12. Optional Graph Refinement

Use graph-derived support only if validation improves performance.

Never propagate matches blindly.

## Validation

Use GroupKFold grouped by `source1_entity_id`.

Track:
- Macro F0.5
- Candidate recall
- Singleton accuracy
- Precision
- Recall

## Outputs

Generate:

- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

Validate with the provided validator before submission.

## Recommended Initial Hyperparameters

| Component | Value |
|-----------|------:|
| Hidden Dimension | 256 |
| Embedding Dimension | 256 |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Batch Size | 128 |
| Learning Rate | 1e-4 |
| JEPA Epochs | 20 |
| LightGBM Trees | 500 |
| TF-IDF Top-K | 30 |
| JEPA Top-K | 30 |

## Success Priorities

1. Candidate Recall
2. Hard Negative Quality
3. Precision Control
4. Threshold Calibration
5. Singleton Detection
6. Address Reasoning
7. JEPA Representations
8. Graph Consistency

The intended design is a **hybrid retrieval + supervised matching pipeline** where JEPA improves representations and retrieval, while gradient boosting makes the final entity-resolution decisions.
