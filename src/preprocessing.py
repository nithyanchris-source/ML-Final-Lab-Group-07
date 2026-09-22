import re
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_RAW = PROCESSED_DIR / "train_raw.csv"
TEST_RAW = PROCESSED_DIR / "test_raw.csv"

TRAIN_CLEAN = PROCESSED_DIR / "train_clean.csv"
TEST_CLEAN = PROCESSED_DIR / "test_clean.csv"

TRAIN_TFIDF = PROCESSED_DIR / "train_tfidf.npz"
TEST_TFIDF = PROCESSED_DIR / "test_tfidf.npz"

TRAIN_LABELS = PROCESSED_DIR / "train_labels.csv"
TEST_LABELS = PROCESSED_DIR / "test_labels.csv"

VECTORIZER_FILE = PROCESSED_DIR / "tfidf_vectorizer.joblib"
FEATURE_NAMES_FILE = PROCESSED_DIR / "tfidf_features.csv"
AUDIT_FILE = PROCESSED_DIR / "data_quality_audit.csv"


def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data():
    if not TRAIN_RAW.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_RAW}")
    if not TEST_RAW.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_RAW}")

    train_df = pd.read_csv(TRAIN_RAW)
    test_df = pd.read_csv(TEST_RAW)

    required_columns = {"text", "label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError("Training data must contain 'text' and 'label' columns.")
    if not required_columns.issubset(test_df.columns):
        raise ValueError("Test data must contain 'text' and 'label' columns.")

    return train_df, test_df


def clean_data(train_df, test_df):
    audit = {}

    audit["Initial train rows"] = len(train_df)
    audit["Initial test rows"] = len(test_df)

    train_missing = train_df["text"].isna().sum()
    test_missing = test_df["text"].isna().sum()

    audit["Train missing text"] = train_missing
    audit["Test missing text"] = test_missing

    train_df = train_df.dropna(subset=["text"]).copy()
    test_df = test_df.dropna(subset=["text"]).copy()

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    train_empty = (train_df["text"].str.strip() == "").sum()
    test_empty = (test_df["text"].str.strip() == "").sum()

    audit["Train empty text removed"] = train_empty
    audit["Test empty text removed"] = test_empty

    train_df = train_df[train_df["text"].str.strip() != ""].copy()
    test_df = test_df[test_df["text"].str.strip() != ""].copy()

    train_duplicates = train_df["text"].duplicated().sum()
    test_duplicates = test_df["text"].duplicated().sum()

    audit["Train duplicates removed"] = train_duplicates
    audit["Test duplicates removed"] = test_duplicates

    train_df = train_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    test_df = test_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    train_texts = set(train_df["text"])
    overlap_count = test_df["text"].isin(train_texts).sum()

    audit["Train-test overlaps removed from test"] = overlap_count

    if overlap_count > 0:
        test_df = test_df[~test_df["text"].isin(train_texts)].reset_index(drop=True)

    remaining_train_duplicates = train_df["text"].duplicated().sum()
    remaining_test_duplicates = test_df["text"].duplicated().sum()
    remaining_overlap = len(set(train_df["text"]) & set(test_df["text"]))

    audit["Remaining train duplicates"] = remaining_train_duplicates
    audit["Remaining test duplicates"] = remaining_test_duplicates
    audit["Remaining train-test overlap"] = remaining_overlap

    if remaining_train_duplicates != 0:
        raise ValueError("Data quality error: duplicate reviews remain in training data.")
    if remaining_test_duplicates != 0:
        raise ValueError("Data quality error: duplicate reviews remain in test data.")
    if remaining_overlap != 0:
        raise ValueError("DATA LEAKAGE DETECTED: train/test reviews overlap.")

    audit["Final train rows"] = len(train_df)
    audit["Final test rows"] = len(test_df)

    audit["Train positive reviews"] = int((train_df["label"] == 1).sum())
    audit["Train negative reviews"] = int((train_df["label"] == 0).sum())
    audit["Test positive reviews"] = int((test_df["label"] == 1).sum())
    audit["Test negative reviews"] = int((test_df["label"] == 0).sum())

    return train_df, test_df, audit


def create_tfidf(train_df, test_df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype="float32",
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    return X_train, X_test, vectorizer


def save_outputs(train_df, test_df, X_train, X_test, vectorizer, audit):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_CLEAN, index=False)
    test_df.to_csv(TEST_CLEAN, index=False)

    save_npz(TRAIN_TFIDF, X_train)
    save_npz(TEST_TFIDF, X_test)

    train_df[["label"]].to_csv(TRAIN_LABELS, index=False)
    test_df[["label"]].to_csv(TEST_LABELS, index=False)

    joblib.dump(vectorizer, VECTORIZER_FILE)

    pd.DataFrame(
        {"feature": vectorizer.get_feature_names_out()}
    ).to_csv(FEATURE_NAMES_FILE, index=False)

    audit["TF-IDF features"] = X_train.shape[1]
    audit["Training matrix rows"] = X_train.shape[0]
    audit["Training matrix columns"] = X_train.shape[1]
    audit["Testing matrix rows"] = X_test.shape[0]
    audit["Testing matrix columns"] = X_test.shape[1]
    audit["Training non-zero values"] = X_train.nnz
    audit["Testing non-zero values"] = X_test.nnz
    audit["TF-IDF fitted on"] = "Training data only"
    audit["Test transformation"] = "Training-fitted vectorizer"
    audit["TF-IDF data type"] = "float32 (configured)"
    audit["Feature matrix format"] = "Sparse CSR / NPZ"

    pd.DataFrame(
        list(audit.items()), columns=["metric", "value"]
    ).to_csv(AUDIT_FILE, index=False)


def main():
    print("=" * 65)
    print("IMDb DATA ENGINEERING PIPELINE")
    print("=" * 65)

    print("\n[1/4] Loading raw data...")
    train_df, test_df = load_data()
    print(f"Initial train rows: {len(train_df):,}")
    print(f"Initial test rows : {len(test_df):,}")

    print("\n[2/4] Cleaning data and preventing leakage...")
    train_df, test_df, audit = clean_data(train_df, test_df)
    print(f"Final train rows: {len(train_df):,}")
    print(f"Final test rows : {len(test_df):,}")
    print("Train/test overlap: 0")

    print("\n[3/4] Creating TF-IDF features...")
    X_train, X_test, vectorizer = create_tfidf(train_df, test_df)
    print(f"Vocabulary size: {X_train.shape[1]:,}")
    print(f"Train matrix: {X_train.shape}")
    print(f"Test matrix : {X_test.shape}")
    print(f"Train non-zero values: {X_train.nnz:,}")
    print(f"Test non-zero values : {X_test.nnz:,}")

    print("\n[4/4] Saving processed outputs...")
    save_outputs(train_df, test_df, X_train, X_test, vectorizer, audit)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated files:")
    for file in [
        TRAIN_CLEAN,
        TEST_CLEAN,
        TRAIN_TFIDF,
        TEST_TFIDF,
        TRAIN_LABELS,
        TEST_LABELS,
        VECTORIZER_FILE,
        FEATURE_NAMES_FILE,
        AUDIT_FILE,
    ]:
        print(f"  ✓ {file.name}")


if __name__ == "__main__":
    main()
