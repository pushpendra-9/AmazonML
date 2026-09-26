# RTX 6000 Pipeline Execution Guide

This document contains the exact steps to spin up the optimized AmazonML Entity Resolution pipeline on your high-performance RTX 6000 GPU workstation.

## 1. Clone the Repository
Pull the optimized, uncapped pipeline directly from your GitHub repository:
```bash
git clone https://github.com/pushpendra-9/AmazonML.git
cd AmazonML
```

## 2. Prepare the Datasets
Because the 1.2GB dataset was too large for GitHub, you must manually place the `dataset/` directory inside `student_resource/`.

Ensure the exact directory structure looks like this before proceeding:
```
AmazonML/
├── student_resource/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv
│   │   │   ├── train_source2.tsv
│   │   │   ├── train_source3.tsv
│   │   │   └── train_ground_truth.tsv
│   │   └── test/
│   │       ├── test_source1.tsv
│   │       ├── test_source2.tsv
│   │       └── test_source3.tsv
│   ├── code/
│   └── output/
```

## 3. Set Up the Environment
Create an isolated Python environment and install the heavily accelerated data-science stack.
```bash
cd student_resource/code/business_entity_resolution
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Note: Because the `requirements.txt` installs PyTorch, FAISS, LightGBM, and RapidFuzz, this will automatically pull the CUDA-compiled binaries on a Linux machine with NVIDIA drivers).*

## 4. Run the Pipeline
The `main.py` script has been fully unlocked. It will:
- Ingest **100%** of the Training set.
- Train the PyTorch **JEPA model for 10 full epochs** directly on the RTX 6000 Tensor cores (`batch_size=1024`).
- Generate **20 candidates per entity** across the entire dataset.
- Train **LightGBM natively on the GPU**.
- Aggressively garbage collect RAM between phases to prevent system bottlenecks.
- Predict and output matches for **100%** of the blind Test set.

Execute the pipeline from the `src/` directory:
```bash
cd src
python3 main.py
```

## 5. Package for Submission
After the pipeline finishes running, your final predictions will be automatically dumped into `student_resource/output/`.

To package your final submission payload for the ML Challenge:
```bash
cd ../../  # Navigate back to student_resource/
zip -r antigravity_submission.zip output code Documentation_template.md
```
Upload `antigravity_submission.zip` to the submission portal!
