# Implement a Competition-Grade JEPA + Gradient-Boosted Entity Resolution System

You are implementing an end-to-end machine-learning solution for a business Entity Resolution competition.

## Objective

Given three independent sources:

- `Source 1`: deduplicated reference entities
- `Source 2`: noisy business records
- `Source 3`: noisy business records

learn to predict, for every Source 1 entity, all matching Source 2 and Source 3 records.

A Source 1 entity may match:

- zero records
- one record
- multiple records

The competition metric is macro-averaged `F0.5`, where precision is weighted more heavily than recall.

The implementation must use **only the supplied training data**. Do not use external databases, APIs, geocoding services, web lookup, business registries, pretrained business/entity databases, or external data augmentation.

The final model must comply with the competition's requirement that the submitted ML model be MIT- or Apache-2.0-licensed and <=8B parameters. Verify licenses for every pretrained component actually used.

---

# 1. Required project structure

Create:

```text
code/
└── business_entity_resolution/
    ├── src/
    │   ├── config.py
    │   ├── io_utils.py
    │   ├── normalization.py
    │   ├── address_features.py
    │   ├── blocking.py
    │   ├── jepa_model.py
    │   ├── jepa_dataset.py
    │   ├── pair_features.py
    │   ├── hard_negative_mining.py
    │   ├── matcher.py
    │   ├── calibration.py
    │   ├── graph_consistency.py
    │   ├── validation.py
    │   └── train_and_predict.py
    ├── README.md
    └── requirements.txt
output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
Documentation_template.md
```

The complete pipeline must be reproducible from the supplied train/test directories.

---

# 2. Data loading

Read all TSV files explicitly with:

```python
pd.read_csv(path, sep="\t")
```

Training:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
```

Testing:

```text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Never assume that country is limited to `US` or `India`.

Treat country as an arbitrary categorical/string value because the test set contains France.

Infer source from file and/or entity ID prefix.

---

# 3. Overall architecture

Implement the following hybrid architecture:

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
                  source-specific LightGBM
                     / XGBoost classifier
                              │
                       probability
                              │
                     probability calibration
                              │
                    singleton/abstention gate
                              │
                  optional graph consistency
                              │
                         final matches
```

The JEPA model is primarily used for:

1. representation learning
2. neural candidate retrieval
3. pairwise similarity features

Do **not** rely on JEPA similarity alone for the final match decision.

---

# 4. Text normalization

Implement robust deterministic normalization.

For every name/address:

1. Unicode normalize.
2. Convert to lowercase.
3. Normalize whitespace.
4. Normalize punctuation.
5. Normalize common symbols.
6. Normalize `&` and `and`.
7. Preserve a raw version.
8. Preserve a lightly normalized version.
9. Create an aggressively normalized alphanumeric version.

For business names, also create a version with common legal suffixes removed.

Examples of suffix concepts:

```text
ltd
limited
llc
inc
incorporated
corp
corporation
company
co
pvt
private
plc
llp
```

Do not blindly remove meaningful tokens. Maintain both suffix-preserved and suffix-removed variants.

Create multiple fields:

```text
name_raw
name_norm
name_core
name_compact

address_raw
address_norm
address_compact
```

Never destroy the original values.

---

# 5. Address component extraction

Create a lightweight deterministic address parser without external lookup.

Extract when present:

```text
house_number
postal_code
city_like_tokens
state_like_tokens
street_like_tokens
district_like_tokens
```

Do not require any component to exist.

Use regex and token-based heuristics only.

For every pair create component agreement features:

```text
postal_exact
house_number_exact
city_token_overlap
state_token_overlap
street_token_overlap
district_token_overlap
```

Also calculate component missingness.

---

# 6. Synthetic corruption engine

Build a corruption engine for JEPA training.

For every training record generate stochastic alternate views using only its existing information.

Corruptions should include:

## Name corruption

- random punctuation removal
- whitespace changes
- abbreviation changes
- legal suffix removal
- token deletion
- token transposition
- character substitution
- character insertion
- character deletion
- repeated-character corruption
- `&` ↔ `and`
- common abbreviation mappings
- casing changes

## Address corruption

- punctuation removal
- token deletion
- component dropout
- comma removal
- whitespace changes
- street abbreviation changes
- postal code dropout
- house-number dropout
- component reordering
- landmark-like token insertion
- character corruption

Do not invent external information.

The purpose is to simulate the noise patterns described by the competition.

---

# 7. JEPA representation-learning design

Implement a small JEPA-style architecture rather than an enormous model.

Use a shared encoder for all business records:

```text
Name tokenizer
      ↓
Transformer encoder
      ↓
name embedding

Address tokenizer
      ↓
Transformer encoder
      ↓
address embedding

Country embedding
      ↓

            fusion MLP
                ↓
          entity embedding z
```

Recommended initial dimensions:

```text
hidden_dim = 256
embedding_dim = 256
transformer_layers = 4
attention_heads = 4 or 8
dropout = 0.1
```

Keep the model comfortably below the 8B parameter restriction.

Use character/subword-friendly tokenization because names and addresses contain typos, abbreviations, transliterations, and unusual tokens.

Prefer a compact custom tokenizer or a permitted MIT/Apache-licensed tokenizer/model.

---

# 8. JEPA predictor

Use:

```text
context encoder
target encoder
predictor
```

For an entity record, create two corrupted views:

```text
view_a → context encoder → z_context
view_b → target encoder  → z_target
```

Then:

```text
z_context → predictor → z_predicted
```

Optimize prediction of target representation.

The target encoder may use an EMA update if desired.

Normalize embeddings before similarity calculations.

---

# 9. Cross-source supervised representation learning

Use ground truth to create positive cross-source pairs.

For every ground-truth relationship:

```text
S1 ↔ S2
S1 ↔ S3
```

encourage their embeddings to be close.

Add a contrastive objective.

Positive:

```text
z_s1 ↔ z_s2
```

Negative:

```text
z_s1 ↔ z_other
```

Use in-batch negatives plus mined hard negatives.

The total JEPA training objective should be conceptually:

```text
L_total =
    lambda_jepa * L_jepa
  + lambda_contrastive * L_contrastive
  + lambda_pair * L_pair
```

Start with approximately:

```text
lambda_jepa = 1.0
lambda_contrastive = 1.0
lambda_pair = 0.5
```

and make weights configurable.

The pair head should predict whether two records represent the same entity.

---

# 10. Hard-negative mining

This is mandatory.

Do not train only against random negatives.

Generate hard negatives from records that have high similarity but are not ground-truth matches.

Sources of hard negatives:

```text
high TF-IDF name similarity
high address similarity
high RapidFuzz similarity
high JEPA cosine similarity
same country + similar name
same city-like token + similar name
same postal code + conflicting business identity
```

Exclude true matches from negatives.

Use progressively harder negatives:

```text
epoch 1:
mostly random + moderately hard

later epochs:
high lexical similarity
high JEPA similarity
near-duplicate branches
```

Ensure the negative sampler is deterministic under a supplied random seed.

---

# 11. Candidate generation

Implement multiple independent retrieval systems.

## Blocker A: exact normalized name

Retrieve records with:

```text
name_norm exact
name_core exact
name_compact exact
```

## Blocker B: exact address components

Retrieve based on combinations such as:

```text
country + postal
country + house_number
country + postal + name token
```

Only use fields when present.

## Blocker C: rare-token blocking

Compute document frequency over each source.

Use rare business-name/address tokens to generate candidates.

Do not give common tokens much blocking power.

## Blocker D: character TF-IDF retrieval

Build character n-gram TF-IDF indexes.

Use ngrams approximately in:

```text
3-grams through 5-grams
```

Retrieve top-k.

Do this independently for:

```text
name
address
name + address
```

## Blocker E: JEPA ANN retrieval

Build an approximate nearest-neighbor index over source 2 and source 3 entity embeddings.

Retrieve top-k for every source 1 entity.

Use cosine similarity.

---

# 12. Candidate union

For every S1:

```python
candidates =
    exact_name_candidates
    | address_candidates
    | tfidf_name_candidates
    | tfidf_address_candidates
    | tfidf_combined_candidates
    | jepa_candidates
```

Deduplicate candidates.

Keep source information.

The final candidate set is the exact set that the matcher receives.

Persist this set to:

```text
output/candidate_pairs.tsv
```

Do not perform additional candidate filtering after this point without updating the file.

Every final prediction MUST be a member of this candidate set.

---

# 13. Candidate-count controls

Candidate recall is more important than aggressive early filtering.

Start with approximate retrieval sizes:

```text
TF-IDF name:       top 30
TF-IDF address:    top 30
TF-IDF combined:   top 30
JEPA:              top 30
exact/block rules: all matches subject to a reasonable cap
```

Then tune these values using grouped validation.

Target an initial final candidate set of roughly:

```text
50–200 candidates per S1
```

but do not force this range if validation shows that a different number improves F0.5.

Measure:

```text
candidate recall
average candidates/S1
95th percentile candidates/S1
maximum candidates/S1
blocking reduction ratio
```

---

# 14. Pair feature engineering

For each candidate `(S1, candidate)` generate a rich feature vector.

## Name features

Include:

```text
exact raw equality
exact normalized equality
exact core equality
Levenshtein normalized similarity
Jaro-Winkler similarity
RapidFuzz ratio
RapidFuzz token_sort_ratio
RapidFuzz token_set_ratio
partial_ratio
character TF-IDF cosine
word TF-IDF cosine
common-token count
Jaccard token similarity
containment similarity
length difference
length ratio
rare-token overlap
```

## Address features

Include analogous metrics for:

```text
raw address
normalized address
compact address
character TF-IDF
word TF-IDF
token Jaccard
edit similarity
```

## Address component features

Include:

```text
postal exact
postal present both
house number exact
house number conflict
city token similarity
state token similarity
street token similarity
district token similarity
number of shared components
number of conflicting components
```

## Country

Include:

```text
country_exact
country_missingness
```

Do not hard-code country values.

## JEPA features

Include:

```text
name_embedding_cosine
address_embedding_cosine
combined_embedding_cosine
full_entity_embedding_cosine
embedding_l2_distance
```

## Cross-features

Include:

```text
name_similarity * address_similarity
name_similarity * jepa_similarity
address_similarity * jepa_similarity
min(name_similarity, address_similarity)
max(name_similarity, address_similarity)
```

Also include missingness indicators.

Aim for approximately 50–150 useful features rather than hundreds of redundant features.

---

# 15. Source-specific matching models

Train separate models:

```text
Matcher_S2:
    S1 ↔ S2

Matcher_S3:
    S1 ↔ S3
```

Use LightGBM as the first-choice model.

XGBoost may be used as an alternative/ensemble.

The target is:

```text
1 = ground-truth match
0 = non-match
```

Train only on candidates produced by the same blocking pipeline.

Do not train on an artificial candidate distribution that differs radically from inference.

---

# 16. Pairwise model training

Use grouped cross-validation by Source 1 entity.

For example:

```text
GroupKFold(n_splits=5)
group = source1_entity_id
```

No S1 entity may appear in both training and validation folds.

For each fold:

1. fit blocking indexes using training data only where leakage matters
2. generate training candidates
3. generate validation candidates
4. create positive/negative labels
5. train matcher
6. score validation candidates
7. optimize the final decision threshold on validation

Store out-of-fold predictions.

---

# 17. Prevent leakage

Do not leak validation ground-truth relationships into candidate generation.

All learned retrieval statistics such as:

```text
TF-IDF vocabulary
document frequencies
rare-token frequencies
embedding indexes
model parameters
calibration parameters
```

must be fitted on the training fold only during cross-validation.

For final test inference, fit on the full training data.

---

# 18. Calibration

Because the competition metric is precision-heavy, raw LightGBM probabilities are not enough.

Implement:

```text
Platt scaling
or
isotonic regression
```

using out-of-fold predictions.

Calibrate S2 and S3 independently.

Then optimize:

```text
threshold_S2
threshold_S3
```

directly for macro F0.5.

Search a dense threshold grid, for example:

```text
0.50 to 0.99
```

and retain the threshold producing the best validation macro F0.5.

Do not optimize ROC-AUC as the primary objective.

---

# 19. Singleton / abstention model

Implement a second-level gate.

For each S1 calculate aggregate candidate evidence:

```text
max pair probability
top-1 probability
top-2 probability
gap between top-1 and top-2
number of candidates above moderate threshold
maximum name similarity
maximum address similarity
maximum JEPA similarity
```

Train a lightweight classifier to estimate:

```text
P(S1 has at least one true match)
```

using grouped training data.

At inference:

```text
if P(any_match) is very low:
    return no matches
else:
    apply pairwise matcher thresholds
```

Tune this gate for macro F0.5.

Be conservative because a false prediction on a true singleton yields zero for that entity.

---

# 20. Avoid forcing one-to-one assignments

Do not use Hungarian matching or any method that forces one candidate per S1.

The correct output can be:

```text
S1 → zero matches
S1 → one match
S1 → multiple S2 matches
S1 → multiple S3 matches
```

Treat candidate decisions independently, subject to optional high-confidence consistency logic.

---

# 21. Optional graph consistency layer

Implement this as an optional feature generator, not a mandatory hard rule.

Create a graph from:

```text
S1 nodes
S2 nodes
S3 nodes
```

For candidate edges, store model probability.

For an S1 candidate, calculate features such as:

```text
maximum supporting two-hop path
S1-S2-S3 support
number of high-confidence neighboring edges
best cross-source embedding agreement
```

Only enable graph-derived signals if cross-validation improves F0.5.

Never blindly propagate a high-confidence match to all connected records.

---

# 22. Optional ensemble

Evaluate:

```text
Model A = LightGBM lexical/address features
Model B = LightGBM + JEPA features
Model C = XGBoost + JEPA features
```

Combine calibrated probabilities only if grouped validation improves macro F0.5.

Example:

```text
p_final =
    alpha * p_lightgbm
  + (1-alpha) * p_xgboost
```

Optimize alpha on validation.

Do not add ensemble complexity unless it improves F0.5.

---

# 23. Threshold strategy

Do not use one universal threshold automatically.

Tune independently:

```text
threshold_S2
threshold_S3
```

Also evaluate conditional thresholds based on evidence.

Example configurable rule:

```text
if normalized_name_exact and strong_address_evidence:
    accept at lower threshold
elif name_high_but_address_conflicts:
    require very high threshold
else:
    normal threshold
```

Do not manually hard-code arbitrary rules unless validation demonstrates improvement.

All thresholds must be configurable in `config.py`.

---

# 24. Evaluation

Implement the exact competition metric:

```python
F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
```

Compute it per Source 1 entity and then macro-average across entities.

Handle:

```text
true set empty
predicted set empty
```

correctly.

In particular:

```text
true empty + prediction empty → F0.5 = 1.0
true empty + prediction non-empty → F0.5 = 0.0
```

Also report:

```text
macro F0.5
micro precision
micro recall
macro precision
macro recall
singleton accuracy
S2 F0.5
S3 F0.5
candidate recall
average candidates/S1
```

---

# 25. Error analysis

After every validation run generate an error-analysis table containing:

```text
S1 ID
candidate ID
true/false
model probability
name similarity
address similarity
JEPA similarity
country agreement
postal agreement
house-number agreement
```

Sort by:

```text
false-positive probability descending
false-negative confidence descending
```

Use this to identify:

- same-name branches
- duplicated addresses
- transliteration failures
- abbreviations
- legal suffix issues
- missing addresses
- country-specific patterns

---

# 26. Ablation experiments

Automatically run and report:

```text
1. exact blocking only
2. exact + fuzzy blocking
3. + TF-IDF retrieval
4. + JEPA retrieval
5. + lexical/address pair features
6. + JEPA features
7. + hard negatives
8. + calibration
9. + singleton gate
10. + graph features
```

For each experiment report:

```text
candidate recall
macro F0.5
precision
recall
average candidates/S1
```

The pipeline should make it easy to identify whether JEPA actually improves the competition metric.

---

# 27. Computational requirements

The implementation must scale beyond toy examples.

Use:

```text
sparse TF-IDF matrices
batched JEPA inference
ANN retrieval instead of all-pairs embedding comparison
vectorized feature calculations where practical
```

Never calculate a full:

```text
S1 × S2
S1 × S3
```

Cartesian similarity matrix if the dataset is large.

Only calculate pair features for blocked candidates.

---

# 28. Reproducibility

Set explicit seeds everywhere practical.

Provide configuration for:

```text
random seed
JEPA dimensions
learning rate
batch size
epochs
temperature
retrieval top-k
LightGBM parameters
negative sampling ratio
thresholds
```

Save:

```text
trained JEPA model
TF-IDF artifacts
matcher models
calibrator models
selected thresholds
configuration
```

where appropriate.

---

# 29. Required inference procedure

Final inference must perform:

```text
1. Load train data
2. Train/finalize normalization
3. Train/finalize JEPA representation model
4. Build S2/S3 embeddings
5. Build TF-IDF indexes
6. Build ANN indexes
7. Generate candidates for every test S1
8. Save EXACT inference candidate set
9. Generate pair features
10. Apply S2 matcher
11. Apply S3 matcher
12. Apply calibration
13. Apply singleton gate
14. Apply optional validated graph consistency
15. Produce final matches
16. Validate output
```

Every test S1 must appear exactly once in both output files.

---

# 30. `candidate_pairs.tsv`

Write:

```text
source1_entity_id
candidate_entity_ids
```

with one row for every test Source 1 entity.

Rules:

- candidate IDs must be valid test S2/S3 IDs
- no duplicates
- empty string for no candidates
- every final match must occur in the candidate list

This file must represent the actual candidate set passed to the matching model.

---

# 31. `matching_results.tsv`

Write:

```text
source1_entity_id
matched_entity_ids
```

with one row for every test Source 1 entity.

Rules:

- every test S1 appears exactly once
- only S2/S3 IDs
- no duplicate IDs
- empty string when there are no matches
- final matches must be a subset of candidates

Preserve deterministic ordering, preferably sorted by source and ID.

---

# 32. Submission validation

Before finishing inference, execute:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Do not finish unless validation passes.

If validation fails, automatically report and correct formatting issues.

---

# 33. Final README

Write exact commands for:

```text
install dependencies
train
cross-validate
run test inference
validate submission
package submission
```

Example:

```bash
pip install -r requirements.txt

python -m src.train_and_predict \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output
```

Document CPU/GPU expectations.

---

# 34. Requirements and licensing

Pin dependency versions in:

```text
requirements.txt
```

Avoid unnecessary dependencies.

Do not use external APIs or internet services.

Before finalizing, explicitly document the license of:

- JEPA implementation
- tokenizer
- pretrained model, if any
- embedding model, if any
- LightGBM/XGBoost
- ANN library

Do not use a pretrained model whose license is incompatible with the competition.

---

# 35. Desired implementation philosophy

Prioritize leaderboard performance in this order:

```text
1. Candidate recall
2. Hard-negative quality
3. False-positive control
4. Threshold calibration
5. Singleton handling
6. Address-component reasoning
7. JEPA representation quality
8. Graph consistency
```

Do not assume a larger neural network is better.

The intended winning strategy is a **hybrid retrieval + supervised pair-classification system** where JEPA provides noise-robust representations but the final decision is made from rich evidence.

---

# 36. Initial hyperparameters to implement

Start with:

```text
JEPA:
hidden_dim=256
embedding_dim=256
layers=4
heads=8
dropout=0.1
batch_size=128
learning_rate=1e-4
epochs=20
temperature=0.07

Retrieval:
TFIDF_name_topk=30
TFIDF_address_topk=30
TFIDF_combined_topk=30
JEPA_topk=30

Matcher:
LightGBM
n_estimators=500
learning_rate=0.03
num_leaves=31
feature_fraction=0.8
bagging_fraction=0.8
```

These are starting values only. Tune them using grouped cross-validation.

---

# 37. Critical success criterion

Do not declare success because pairwise accuracy or ROC-AUC is high.

The main optimization target is:

```text
GROUPED VALIDATION MACRO F0.5
```

with special attention to:

```text
false merges
true singletons
same-name different-branch entities
```

A more conservative model with slightly lower recall but substantially fewer false positives is often preferable under this metric.

---

# 38. Deliverable

At completion, provide:

```text
output/matching_results.tsv
output/candidate_pairs.tsv

code/business_entity_resolution/src/*
code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt

Documentation_template.md
```

Also report:

```text
best validation F0.5
candidate recall
average candidate count
S2 threshold
S3 threshold
singleton-gate threshold
JEPA ablation improvement
hard-negative improvement
```

Finally, package everything into:

```text
<team_name>_submission.zip
```

with the required directory structure.

The implementation must be runnable end-to-end from the supplied data without external data lookup.