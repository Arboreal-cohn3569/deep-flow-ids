import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_plot(model, X_test, y_test, class_names, save_dir="assets"):
    """
    Computes classification reports and saves normalized confusion matrices to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n==================== Evaluation: {model.name} ====================")
    
    y_pred_probs = model.predict(X_test, batch_size=512)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(f'Normalized Confusion Matrix - {model.name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{model.name}_cm.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Confusion matrix plot saved: {plot_path}")