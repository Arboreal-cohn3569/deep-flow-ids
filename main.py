import argparse
from src.data_loader import load_and_preprocess_data
from src.models import build_1d_cnn, build_rnn_lstm, build_hybrid_cnn_lstm
from src.train import train_model
from src.evaluate import evaluate_and_plot

def main():
    parser = argparse.ArgumentParser(description="DeepFlow-IDS Training and Evaluation Pipeline")
    parser.add_argument("--data_path", type=str, default="data/raw/*.csv", help="Glob pattern for CSV files")
    parser.add_argument("--samples_per_class", type=int, default=10000, help="Downsampling threshold per class")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch count")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation ratio")
    args = parser.parse_args()

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data(
        raw_data_path=args.data_path,
        sample_per_class=args.samples_per_class
    )
    
    input_shape = (X_train.shape[1], 1)
    num_classes = len(class_names)

    # Step 2: Model list
    pipeline = [
        build_1d_cnn(input_shape, num_classes),
        build_rnn_lstm(input_shape, num_classes),
        build_hybrid_cnn_lstm(input_shape, num_classes)
    ]

    # Step 3: Run pipeline
    for model in pipeline:
        trained_model, _ = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split
        )
        evaluate_and_plot(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names
        )

if __name__ == "__main__":
    main()