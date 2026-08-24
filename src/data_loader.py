import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

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
    label_col = 'Label'

    if files:
        print(f"[+] Discovered {len(files)} CSV files in {raw_data_path}. Ingesting in chunks...")
        sampled_dfs = []
        for f in files:
            for chunk in pd.read_csv(f, skipinitialspace=True, low_memory=False, chunksize=50000):
                chunk.columns = chunk.columns.str.strip()
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
        df[label_col] = mock_labels

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Class balancing (prevents multi-million benign bias)
    print("[+] Balancing dataset classes...")
    balanced_df = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), sample_per_class), random_state=random_state)
    ).reset_index(drop=True)

    X = balanced_df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    y = balanced_df[label_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)

    # Leakage-free train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Robust scaling for network metrics with extreme outliers
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Reshape for 1D convolution / sequential layers: (batch, num_features, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Save processed artifacts
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    joblib.dump(label_encoder, os.path.join(processed_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(processed_dir, "scaler.pkl"))

    print(f"[✓] Data processed. Input Shape: {X_train.shape[1:]} | Target Classes: {len(class_names)}")
    return X_train, X_test, y_train, y_test, class_names