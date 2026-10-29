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
    title=None,
    decimals=1,
    dpi=200,
    save_dpi=300,
    figsize=(7, 4.5),
    save_path=None,
    facecolor="white",
):
    """
    Create a 2D or 3D scatter plot with optional fit line.

    fit_line:
        False        — points only
        "linear"     — OLS of y on x  (or a plane in 3D)
        "quadratic"  — OLS of y on x and x²  (2D only)
    """
    if isinstance(x, str):
        x = [x]

    if len(x) == 1:
        ax = _scatter_2d(
            df, y, x[0], fit_line=fit_line, title=title, decimals=decimals,
            dpi=dpi, figsize=figsize, facecolor=facecolor,
        )
    else:
        ax = _scatter_3d(
            df, y, x[0], x[1], fit_line=fit_line, title=title, decimals=decimals,
            dpi=dpi, figsize=figsize, facecolor=facecolor,
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


def _fit_kind(fit_line):
    if not fit_line:
        return None
    if fit_line == "linear":
        return "linear"
    if fit_line == "quadratic":
        return "quadratic"
    raise ValueError("fit_line must be False, 'linear', or 'quadratic'")


def _numeric_cols(df, *names):
    """Return aligned 1D float arrays, even if the same column is used twice."""
    series = [pd.to_numeric(df[name], errors="coerce") for name in names]
    mask = series[0].notna()
    for s in series[1:]:
        mask &= s.notna()
    return [s.loc[mask].to_numpy(dtype=float) for s in series]


def _style_2d(ax, decimals):
    fmt = FuncFormatter(lambda v, _: f"{v:,.{decimals}f}")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(labelsize=10, length=5, width=1.1)
    ax.ticklabel_format(style="plain")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def _scatter_2d(df, y, x_var, fit_line, title, decimals, dpi, figsize, facecolor):
    xv, yv = _numeric_cols(df, x_var, y)
    if xv.size == 0:
        raise ValueError("No data to plot after dropping NaNs.")

    corr = float(pd.Series(xv).corr(pd.Series(yv)))
    if not title:
        title = f"{y} and {x_var}\nCorrelation: {corr:.2f}"

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

    kind = _fit_kind(fit_line)
    if kind is not None and len(xv) >= (3 if kind == "quadratic" else 2):
        degree = 2 if kind == "quadratic" else 1
        coefs = np.polyfit(xv, yv, degree)
        x_line = np.linspace(xv.min(), xv.max(), 200)
        y_line = np.polyval(coefs, x_line)
        ax.plot(
            x_line,
            y_line,
            color="#c0392b",
            linewidth=1.8,
            zorder=4,
        )

    _style_2d(ax, decimals)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(y, fontsize=11)
    return ax


def _scatter_3d(df, y, x_var, z_var, fit_line, title, decimals, dpi, figsize, facecolor):
    kind = _fit_kind(fit_line)
    if kind == "quadratic":
        raise ValueError("fit_line='quadratic' is only available for 2D scatter plots.")

    xv, zv, yv = _numeric_cols(df, x_var, z_var, y)
    if xv.size == 0:
        raise ValueError("No data to plot after dropping NaNs.")

    if not title:
        title = f"{y}, {x_var}, and {z_var}"

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax = fig.add_subplot(111, projection="3d", facecolor=facecolor)

    ax.scatter(
        xv,
        zv,
        yv,
        s=32,
        c="#7eb6d9",
        alpha=0.78,
        edgecolors="#1f1f1f",
        linewidths=0.45,
        depthshade=True,
    )

    if kind == "linear":
        X_data = np.column_stack([xv, zv, np.ones(len(xv))])
        coeffs, *_ = np.linalg.lstsq(X_data, yv, rcond=None)
        a, b, c = coeffs

        x_surf = np.linspace(xv.min(), xv.max(), 12)
        z_surf = np.linspace(zv.min(), zv.max(), 12)
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

    fmt = FuncFormatter(lambda v, _: f"{v:,.{decimals}f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.zaxis.set_major_formatter(fmt)

    ax.set_title(title, fontsize=13, pad=10)
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
