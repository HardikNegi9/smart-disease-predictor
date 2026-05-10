import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def log_and_save_confusion_matrix(y_true, y_pred, labels, class_names, png_path, logger):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info(f"Confusion Matrix:\n{cm}")

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    logger.info(f"Confusion matrix PNG saved to {png_path}")
