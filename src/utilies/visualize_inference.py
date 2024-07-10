#!/usr/bin/env python3
import os
import scipy
import scipy.stats
import torch

import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

import numpy as np
import seaborn as sns

from sklearn.metrics import mean_squared_error


def visualize_inference(model, model_state, test_loader, save_dir):
    set_matplotlib_formats("pdf")
    plt.rcParams.update({"font.size": 16, "font.weight": "bold"})

    # load modal
    model_state = torch.load(model_state)
    model.load_state_dict(model_state["state_dict"])
    
    # infer
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for input, target in test_loader:
            pred = model(input)
            predictions.append(pred.item())
            targets.append(target.item())

    predictions = np.array(predictions).reshape(-1, 1)
    targets = np.array(targets).reshape(-1, 1)

    pearsonr = scipy.stats.pearsonr(predictions, targets)
    r2 = pearsonr.statistic ** 2
    vmin = np.min([predictions.min(), targets.min()])
    vmax = np.max([predictions.max(), targets.max()])

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.regplot(
        x=targets, y=predictions, ax=ax,
        line_kws={"linestyle": "--", "color": "orange"},
        scatter_kws={"alpha": 0.5, "edgecolors": "steelblue"},
    )
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="red")
    ax.axis("equal")
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_title(f"mse: {mean_squared_error(target, predictions):.2f}, r2: {r2:.2f}")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Predictions")
    plt.tight_layout()

    img_file = os.path.join(save_dir, "inference.pdf")
    plt.savefig(img_file)
    