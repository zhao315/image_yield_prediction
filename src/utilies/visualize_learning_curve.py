#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats


def visualize_learning_curve(train_loss, val_loss, save_dir):
    set_matplotlib_formats("pdf")
    plt.rcParams.update({"font.size": 16, "font.weight":"bold"})

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, linewidth=3, label="training loss")
    plt.plot(val_loss, linewidth=3, label="validation loss")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    img_file = os.path.join(save_dir, "learning_curve.pdf")
    plt.savefig(img_file)


