import matplotlib.pyplot as plt

from src.model.model_constants import DATA_LABELS
from sklearn.metrics import confusion_matrix

import numpy as np
import tempfile

def get_confusion_matrix(predict, correct, return_fig=False):
    labels = sorted(DATA_LABELS.keys())
    n_classes = len(DATA_LABELS)
    conf_mtrx = confusion_matrix(correct, predict, [i for i in range(n_classes)])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(conf_mtrx, cmap="Blues")

    for (i, j), z in np.ndenumerate(conf_mtrx):
        ax.text(
            j,i, 
            '{:0.1f}'.format(z),
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")


    plt.title("Confusion Matirx of Model")
    if return_fig:
        return plt
    else:
        fo = tempfile.NamedTemporaryFile("w+")

        plt.savefig(fo.name, format="png")
        return fo
    