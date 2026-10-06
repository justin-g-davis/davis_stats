import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from .trim import trim


def boxplot(
    series,
    title=None,
    trim_outliers=100,
    dpi=200,
    save_dpi=300,
    figsize=(7, 4.5),
    show_mean=True,
    show_stats=True,
    horizontal=False,
    save_path=None,
    facecolor="white",
):
    if not title:
        title = series.name if getattr(series, "name", None) else "Distribution"

    original = np.asarray(pd.Series(series).dropna(), dtype=float).ravel()
    if original.size == 0:
        raise ValueError("No data to plot after dropping NaNs.")

    data = original.copy()
    if trim_outliers < 100:
        trimmed = trim(
            pd.Series(original, name=getattr(series, "name", None)),
            trim_outliers,
        )
        data = np.asarray(pd.Series(trimmed).dropna(), dtype=float).ravel()
        if data.size == 0:
            raise ValueError("No data to plot after trimming.")
        title = f"{title} (outliers removed at {trim_outliers}% level)"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    value_axis = "x" if horizontal else "y"

    ax.boxplot(
        data,
        vert=not horizontal,
        patch_artist=True,
        widths=0.45,
        whis=1.5,
        boxprops=dict(facecolor="#7eb6d9", edgecolor="#1f1f1f", linewidth=1.6),
        medianprops=dict(color="#c0392b", linewidth=2.2),
        whiskerprops=dict(color="#1f1f1f", linewidth=1.4),
        capprops=dict(color="#1f1f1f", linewidth=1.4),
        flierprops=dict(
            marker="o",
            markersize=4.5,
            markerfacecolor="#7a7a7a",
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.6,
            alpha=0.65,
        ),
    )

    ax.grid(True, axis=value_axis, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.grid(False, axis="x" if value_axis == "y" else "y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    if horizontal:
        ax.set_yticks([1])
        ax.set_yticklabels([""])
    else:
        ax.set_xticks([1])
        ax.set_xticklabels([""])

    ax.ticklabel_format(style="plain", axis=value_axis)
    ax.tick_params(axis=value_axis, labelsize=11, length=5, width=1.1)
    getattr(ax, f"{value_axis}axis").set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:,.2f}")
    )

    q1, med, q3 = np.percentile(original, [25, 50, 75])
    mean = float(np.mean(original))
    iqr = q3 - q1

    if show_mean:
        if horizontal:
            ax.scatter(
                [mean], [1], s=55, c="#1a1a1a", zorder=5,
                marker="D", label=f"Mean: {mean:,.2f}",
            )
        else:
            ax.scatter(
                [1], [mean], s=55, c="#1a1a1a", zorder=5,
                marker="D", label=f"Mean: {mean:,.2f}",
            )
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)

    if show_stats:
        stats = (
            f"n = {len(original):,}\n"
            f"Mean:   {mean:>10,.2f}\n"
            f"Median: {med:>10,.2f}\n"
            f"Q1:     {q1:>10,.2f}\n"
            f"Q3:     {q3:>10,.2f}\n"
            f"IQR:    {iqr:>10,.2f}"
        )
        ax.text(
            0.98, 0.02, stats, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.92,
            ),
        )

    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path, dpi=save_dpi, facecolor=facecolor,
            bbox_inches="tight", pad_inches=0.15,
        )

    plt.show(block=False)
    return ax
