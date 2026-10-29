import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from ..reg_modeling.reg import reg


def residuals(
    df,
    y,
    x,
    dummies=None,
    logistic=False,
    dpi=200,
    save_dpi=300,
    figsize=(7, 4.5),
    save_path=None,
    facecolor="white",
):
    """
    Plot residuals vs fitted values to check linearity.
    """
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    if results is None:
        print("Error: Could not fit regression model")
        return

    fitted = np.asarray(results.fittedvalues, dtype=float)
    resid = np.asarray(results.resid, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    ax.scatter(
        fitted,
        resid,
        s=32,
        c="#7eb6d9",
        alpha=0.78,
        edgecolors="#1f1f1f",
        linewidths=0.45,
        zorder=3,
    )
    ax.axhline(
        0, color="#c0392b", linestyle="--", linewidth=1.8, zorder=4,
    )

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(labelsize=10, length=5, width=1.1)
    ax.ticklabel_format(style="plain")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.set_title("Residuals vs Fitted Values", fontsize=13, pad=10)
    ax.set_xlabel("Fitted Values", fontsize=11)
    ax.set_ylabel("Residuals", fontsize=11)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path, dpi=save_dpi, facecolor=facecolor,
            bbox_inches="tight", pad_inches=0.15,
        )
    plt.show(block=False)
    return ax
