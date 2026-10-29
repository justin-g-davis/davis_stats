import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def scatter(
    df,
    y,
    x,
    fit_line=False,
    dpi=200,
    save_dpi=300,
    figsize=(7, 4.5),
    save_path=None,
    facecolor="white",
):
    """
    Create a 2D or 3D scatter plot with optional fit line.
    """
    if isinstance(x, str):
        x = [x]

    if len(x) == 1:
        ax = _scatter_2d(
            df, y, x[0], fit_line=fit_line, dpi=dpi, figsize=figsize,
            facecolor=facecolor,
        )
    else:
        ax = _scatter_3d(
            df, y, x[0], x[1], fit_line=fit_line, dpi=dpi, figsize=figsize,
            facecolor=facecolor,
        )

    fig = ax.figure
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path, dpi=save_dpi, facecolor=facecolor,
            bbox_inches="tight", pad_inches=0.15,
        )
    plt.show(block=False)
    return ax


def _style_2d(ax):
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


def _scatter_2d(df, y, x_var, fit_line, dpi, figsize, facecolor):
    plot_df = df[[x_var, y]].dropna()
    if plot_df.empty:
        raise ValueError("No data to plot after dropping NaNs.")

    xv = plot_df[x_var].to_numpy(dtype=float)
    yv = plot_df[y].to_numpy(dtype=float)
    corr = float(pd.Series(xv).corr(pd.Series(yv)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    ax.scatter(
        xv,
        yv,
        s=32,
        c="#7eb6d9",
        alpha=0.78,
        edgecolors="#1f1f1f",
        linewidths=0.45,
        zorder=3,
    )

    if fit_line and len(xv) >= 2:
        slope, intercept = np.polyfit(xv, yv, 1)
        x_line = np.array([xv.min(), xv.max()])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#c0392b",
            linewidth=1.8,
            zorder=4,
        )

    _style_2d(ax)
    ax.set_title(f"{y} and {x_var}\nCorrelation: {corr:.3f}", fontsize=13, pad=10)
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(y, fontsize=11)
    return ax


def _scatter_3d(df, y, x_var, z_var, fit_line, dpi, figsize, facecolor):
    plot_df = df[[x_var, z_var, y]].dropna()
    if plot_df.empty:
        raise ValueError("No data to plot after dropping NaNs.")

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax = fig.add_subplot(111, projection="3d", facecolor=facecolor)

    ax.scatter(
        plot_df[x_var],
        plot_df[z_var],
        plot_df[y],
        s=32,
        c="#7eb6d9",
        alpha=0.78,
        edgecolors="#1f1f1f",
        linewidths=0.45,
        depthshade=True,
    )

    if fit_line:
        X_data = np.column_stack(
            [plot_df[x_var], plot_df[z_var], np.ones(len(plot_df))]
        )
        y_data = plot_df[y].to_numpy(dtype=float)
        coeffs, *_ = np.linalg.lstsq(X_data, y_data, rcond=None)
        a, b, c = coeffs

        x_surf = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 12)
        z_surf = np.linspace(plot_df[z_var].min(), plot_df[z_var].max(), 12)
        X_mesh, Z_mesh = np.meshgrid(x_surf, z_surf)
        Y_mesh = a * X_mesh + b * Z_mesh + c
        ax.plot_surface(
            X_mesh,
            Z_mesh,
            Y_mesh,
            color="#c0392b",
            alpha=0.28,
            edgecolor="none",
            shade=False,
        )

    ax.set_title(f"{y}, {x_var}, and {z_var}", fontsize=13, pad=10)
    ax.set_xlabel(x_var, fontsize=11, labelpad=8)
    ax.set_ylabel(z_var, fontsize=11, labelpad=8)
    ax.set_zlabel(y, fontsize=11, labelpad=8)
    ax.tick_params(labelsize=9, length=4, width=1.0)

    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.view_init(elev=25, azim=135)

    grid = (0.55, 0.55, 0.55, 0.35)
    ax.xaxis._axinfo["grid"]["color"] = grid
    ax.yaxis._axinfo["grid"]["color"] = grid
    ax.zaxis._axinfo["grid"]["color"] = grid
    return ax
