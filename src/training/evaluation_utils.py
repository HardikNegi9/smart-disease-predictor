import os
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def log_and_save_confusion_matrix(y_true, y_pred, labels=None, class_names=None, png_path="confusion_matrix.png", logger=None):
    logger = logger or logging
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info(f"Confusion Matrix:\n{cm}")

    out_dir = os.path.dirname(png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    logger.info(f"Confusion matrix PNG saved to {png_path}")
