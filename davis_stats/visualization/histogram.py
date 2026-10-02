import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from math import erf, sqrt

TRUE_1SD, TRUE_2SD, TRUE_3SD = [erf(k / sqrt(2)) * 100 for k in (1, 2, 3)]

def histogram(
    series,
    title=None,
    bins=30,
    trim_outliers=100,
    details=True,
    dpi=160,
    save_dpi=300,
    figsize=(7, 4.5),
    show_normal=False,
    save_path=None,
    facecolor="white",
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    if not title:
        title = series.name if getattr(series, "name", None) else "Distribution"

    original_data = np.asarray(series.dropna(), dtype=float).ravel()
    if original_data.size == 0:
        raise ValueError("No data to plot after dropping NaNs.")

    plot_series = series
    if trim_outliers < 100:
        plot_series = trim(series, trim_outliers)
        title = f"{title} (outliers removed at {trim_outliers}% level)"

    data = np.asarray(plot_series.dropna(), dtype=float).ravel()

    counts, edges, _ = ax.hist(
        data,
        bins=bins,
        edgecolor="#1f1f1f",
        linewidth=0.8,
        color="#7eb6d9",
        alpha=0.85,
    )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(labelsize=10, length=5, width=1.1)
    ax.ticklabel_format(style="plain", axis="x")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    if details:
        # stats always on FULL original sample (pre-trim)
        mean = float(np.mean(original_data))
        median = float(np.median(original_data))
        std = float(np.std(original_data))
        skew_val = float(pd.Series(original_data).skew())

        w1 = np.mean(np.abs(original_data - mean) <= 1 * std) * 100
        w2 = np.mean(np.abs(original_data - mean) <= 2 * std) * 100
        w3 = np.mean(np.abs(original_data - mean) <= 3 * std) * 100

        ax.axvline(mean, color="#c0392b", linestyle="--", linewidth=1.8,
                   label=f"Mean: {mean:,.2f}")
        ax.axvline(median, color="#1e8449", linestyle=":", linewidth=2.0,
                   label=f"Median: {median:,.2f}")
        ax.plot([mean], [0], marker="^", color="#c0392b", markersize=8,
                clip_on=False, zorder=5)
        ax.plot([median], [0], marker="^", color="#1e8449", markersize=8,
                clip_on=False, zorder=5)

        if show_normal and std > 0:
            x = np.linspace(edges[0], edges[-1], 400)
            pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            ax.plot(x, pdf * len(data) * (edges[1] - edges[0]),
                    color="#1f1f1f", linewidth=1.6, alpha=0.85, label="Normal")

        stats_text = (
            f"n = {len(original_data):,}\n"
            f"Mean:        {mean:>12,.2f}\n"
            f"Median:      {median:>12,.2f}\n"
            f"Std Dev:     {std:>12,.2f}\n"
            f"Skewness:    {skew_val:>12,.3f}\n"
            f"Within 1 SD: {w1:>6.1f}%  ({TRUE_1SD:.1f}%)\n"
            f"Within 2 SD: {w2:>6.1f}%  ({TRUE_2SD:.1f}%)\n"
            f"Within 3 SD: {w3:>6.1f}%  ({TRUE_3SD:.1f}%)"
        )
        ax.text(
            0.98, 0.96, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92),
        )
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor=facecolor,
                    bbox_inches="tight", pad_inches=0.15)

    plt.show(block=False)
    return ax
