import os
from tensorflow.keras import callbacks

def train_model(model, X_train, y_train, epochs=15, batch_size=512, val_split=0.15, save_dir="assets"):
    """
    Trains model with EarlyStopping and ReduceLROnPlateau, saving the resulting .keras file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )
    
    print(f"\n==================== Training: {model.name} ====================")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    save_path = os.path.join(save_dir, f"{model.name}.keras")
    model.save(save_path)
    print(f"[✓] Saved model artifact: {save_path}")
    
    return model, history