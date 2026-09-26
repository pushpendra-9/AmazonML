# Business Entity Resolution Pipeline

## Overview
This repository contains a full hybrid pipeline for Business Entity Resolution using a PyTorch-based Joint Embedding Predictive Architecture (JEPA) model, Faiss + TF-IDF blocking, RapidFuzz string feature engineering, and a LightGBM classification head. 

## Requirements
```
pip install -r requirements.txt
```

## How to Run End-to-End
1. Ensure the `dataset/` directory is located at the root of the project (i.e. three directories up from `src/`).
2. Run the main orchestration script from this directory:
```bash
cd src
python3 main.py
```
3. The script will dynamically train the JEPA model, generate candidates, extract features, and train the matcher.
4. Outputs (`matching_results.tsv` and `candidate_pairs.tsv`) will be generated in the `output/` directory (three directories up from `src/`).

## Architecture Details
Please refer to the root `Documentation_template.md` for a full breakdown of the methodology and performance metrics.
