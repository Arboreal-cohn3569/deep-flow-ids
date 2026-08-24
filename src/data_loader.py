import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

def find_label_column(df_columns):
    """Dynamically finds the label column regardless of case or whitespace."""
    for col in df_columns:
        if col.strip().lower() in ['label', 'class', 'attack', 'attack_type']:
            return col
    # Fallback to the last column
    return df_columns[-1]

def load_and_preprocess_data(
    raw_data_path="data/raw/*.csv",
    processed_dir="data/processed",
    sample_per_class=10000,
    test_size=0.2,
    random_state=42
):
    """
    Loads CIC-IDS2017 files in memory-safe chunks, drops inf/nan, applies stratified downsampling,
    performs leakage-free scaling, and caches arrays to disk.
    """
    os.makedirs(processed_dir, exist_ok=True)
    files = glob.glob(raw_data_path)

    if files:
        print(f"[+] Discovered {len(files)} CSV files in {raw_data_path}. Ingesting in chunks...")
        sampled_dfs = []
        for f in files:
            print(f"  -> Reading {os.path.basename(f)}...")
            for chunk in pd.read_csv(f, skipinitialspace=True, low_memory=False, chunksize=50000):
                # Standardize all column names by stripping spaces
                chunk.columns = [col.strip() for col in chunk.columns]
                
                # Standardize label column name to exactly 'Label'
                actual_label_col = find_label_column(chunk.columns)
                if actual_label_col != 'Label':
                    chunk.rename(columns={actual_label_col: 'Label'}, inplace=True)
                
                # Replace Inf with NaN and drop missing rows
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.dropna(inplace=True)
                
                sampled_dfs.append(chunk)
                
        df = pd.concat(sampled_dfs, axis=0, ignore_index=True)
    else:
        print("[!] No CSV files found in data/raw/. Generating synthetic baseline...")
        np.random.seed(random_state)
        n_samples, n_features = 50000, 78
        classes = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'FTP-Patator']
        mock_data = np.random.randn(n_samples, n_features)
        mock_labels = np.random.choice(classes, size=n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        
        feature_cols = [f"Feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(mock_data, columns=feature_cols)
        df['Label'] = mock_labels

    # Standardize column names across entire merged DataFrame
    df.columns = [col.strip() for col in df.columns]
    actual_label_col = find_label_column(df.columns)
    if actual_label_col != 'Label':
        df.rename(columns={actual_label_col: 'Label'}, inplace=True)

    # Clean non-printable / encoding artifacts in labels
    df['Label'] = df['Label'].astype(str).str.strip()
    
    # Class balancing (downsample large classes to prevent benign dominance)
    print("\n[+] Class distribution before balancing:")
    print(df['Label'].value_counts())

    balanced_dfs = []
    for label_val, group in df.groupby('Label'):
        n_samples = min(len(group), sample_per_class)
        balanced_dfs.append(group.sample(n=n_samples, random_state=random_state))
    balanced_df = pd.concat(balanced_dfs, axis=0, ignore_index=True)

    print("\n[+] Class distribution after balancing:")
    print(balanced_df['Label'].value_counts())

    # Separate numeric features from the Label column
    y_raw = balanced_df['Label']
    X_raw = balanced_df.drop(columns=['Label']).select_dtypes(include=[np.number])

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    class_names = [str(cls) for cls in label_encoder.classes_]

    # Leakage-free train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw.values, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Robust scaling (outlier resilient)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Reshape for 1D convolution and temporal layers: (batch, num_features, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Save processed artifacts for fast reuse
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    joblib.dump(label_encoder, os.path.join(processed_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(processed_dir, "scaler.pkl"))

    print(f"\n[✓] Data processed. Input Shape: {X_train.shape[1:]} | Target Classes: {len(class_names)}")
    return X_train, X_test, y_train, y_test, class_names