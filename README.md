# ML-Final-Lab-Group-07

Aspiring Data Analysts who do pure data cleaning, data analysis, finding patterns, model prediction and deployment.

## Data Engineer Contribution

### Dataset

The project uses the **Stanford IMDb Large Movie Review Dataset** for binary sentiment classification.

The labeled dataset contains 25,000 training reviews and 25,000 test reviews before quality cleaning. The original raw IMDb archive is kept outside GitHub and is not modified.

### Data Engineering Pipeline

The automated pipeline is implemented in `src/preprocessing.py`.

It performs:
1. Raw data validation and schema checking.
2. HTML-tag removal and whitespace normalization.
3. Missing and empty-text checks.
4. Duplicate-review removal within each split.
5. Train/test overlap detection and removal from the test set.
6. Leakage validation.
7. TF-IDF feature generation.
8. Saving cleaned data, labels, sparse feature matrices, vectorizer, feature names, and an audit file.

### Data Quality Results

| Metric | Result |
|---|---:|
| Initial training reviews | 25,000 |
| Initial test reviews | 25,000 |
| Training duplicates removed | 98 |
| Test duplicates removed | 201 |
| Train/test overlaps removed from test | 123 |
| Final training reviews | 24,902 |
| Final test reviews | 24,676 |
| Remaining train/test overlap | 0 |
| TF-IDF features | 433,653 |
| Training non-zero TF-IDF values | 7,528,088 |
| Test non-zero TF-IDF values | 6,921,225 |

The complete audit is available at `data/processed/data_quality_audit.csv`.

### Leakage Prevention

The TF-IDF vectorizer is **fitted on training data only** using `fit_transform`.

The test data is then transformed using the same training-fitted vectorizer using `transform`.

The pipeline also verifies that no cleaned review text remains duplicated across the training and test sets.

### Processed Feature Storage

The TF-IDF matrices contain **433,653 features**, so they are stored as sparse CSR matrices in `.npz` format rather than dense CSV files. This avoids an unnecessarily large dense representation while preserving the full feature matrix.

The processed-data folder documents the generated artifacts in `data/processed/README.md`.

### Reproduce the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the generated raw CSV files in:

`data/processed/train_raw.csv`
`data/processed/test_raw.csv`

Then run:

```bash
python src/preprocessing.py
```

The raw CSV copies are intentionally ignored by Git through `.gitignore`.

### Data Engineer Deliverables

- Automated preprocessing pipeline: `src/preprocessing.py`
- Data-quality audit: `data/processed/data_quality_audit.csv`
- Processed-data documentation: `data/processed/README.md`
- Reproducible dependency list: `requirements.txt`
- Raw-data protection: `.gitignore`