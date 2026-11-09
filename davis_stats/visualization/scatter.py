import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def scatter(
    df,
    y,
    x,
    fit_line=False,
    color=False,
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
        "quadratic"  — OLS of y on x and x²  (or quadratic surface in 3D)
        "logistic"   — logit of binary y on x (curve in 2D; surface in 3D)

    color:
        If True, color points by y (blue=low, red=high).
    """
    if isinstance(x, str):
        x = [x]

    if len(x) == 1:
        ax = _scatter_2d(
            df, y, x[0], fit_line=fit_line, color=color, title=title,
            decimals=decimals, dpi=dpi, figsize=figsize, facecolor=facecolor,
        )
    else:
        ax = _scatter_3d(
            df, y, x[0], x[1], fit_line=fit_line, color=color, title=title,
            decimals=decimals, dpi=dpi, figsize=figsize, facecolor=facecolor,
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
    if fit_line in ("linear", "quadratic", "logistic"):
        return fit_line
    raise ValueError("fit_line must be False, 'linear', 'quadratic', or 'logistic'")


def _require_binary_y(yv):
    vals = set(np.unique(yv))
    if not vals.issubset({0.0, 1.0}):
        raise ValueError("fit_line='logistic' requires a binary 0/1 outcome.")
    if len(vals) < 2:
        raise ValueError(
            "fit_line='logistic' needs both 0 and 1 outcomes in the plotted data."
        )


def _numeric_cols(df, *names):
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


def _scatter_points(ax, coords, yv, color, dim=2):
    """coords is (x,) for 2D or (x, z) for 3D."""
    base = dict(
        s=32,
        alpha=0.78,
        edgecolors="#1f1f1f",
        linewidths=0.45,
    )
    if dim == 2:
        if color:
            return ax.scatter(
                coords[0], yv, c=yv, cmap="coolwarm", zorder=3, **base
            )
        return ax.scatter(
            coords[0], yv, c="#7eb6d9", zorder=3, **base
        )

    if color:
        return ax.scatter(
            coords[0], coords[1], yv, c=yv, cmap="coolwarm",
            depthshade=True, **base
        )
    return ax.scatter(
        coords[0], coords[1], yv, c="#7eb6d9",
        depthshade=True, **base
    )


def _scatter_2d(df, y, x_var, fit_line, color, title, decimals, dpi, figsize, facecolor):
    xv, yv = _numeric_cols(df, x_var, y)
    if xv.size == 0:
        raise ValueError("No data to plot after dropping NaNs.")

    corr = float(pd.Series(xv).corr(pd.Series(yv)))
    if not title:
        title = f"{y} and {x_var}\nCorrelation: {corr:.2f}"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    sc = _scatter_points(ax, (xv,), yv, color=color, dim=2)
    if color:
        fig.colorbar(sc, ax=ax, label=y)

    kind = _fit_kind(fit_line)
    min_n = 3 if kind == "quadratic" else 2
    if kind is not None and len(xv) >= min_n:
        x_line = np.linspace(xv.min(), xv.max(), 200)

        if kind == "logistic":
            _require_binary_y(yv)
            fitted = sm.Logit(
                yv, sm.add_constant(xv, has_constant="add")
            ).fit(method="lbfgs", maxiter=5000, disp=0)
            y_line = fitted.predict(sm.add_constant(x_line, has_constant="add"))
        else:
            degree = 2 if kind == "quadratic" else 1
            coefs = np.polyfit(xv, yv, degree)
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


def _scatter_3d(df, y, x_var, z_var, fit_line, color, title, decimals, dpi, figsize, facecolor):
    kind = _fit_kind(fit_line)

    xv, zv, yv = _numeric_cols(df, x_var, z_var, y)
    if xv.size == 0:
        raise ValueError("No data to plot after dropping NaNs.")

    if not title:
        title = f"{y}, {x_var}, and {z_var}"

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    fig.patch.set_facecolor(facecolor)
    ax = fig.add_subplot(111, projection="3d", facecolor=facecolor)

    sc = _scatter_points(ax, (xv, zv), yv, color=color, dim=3)
    if color:
        fig.colorbar(sc, ax=ax, shrink=0.7, label=y)

    if kind in ("linear", "quadratic", "logistic"):
        x_surf = np.linspace(xv.min(), xv.max(), 12 if kind != "quadratic" else 20)
        z_surf = np.linspace(zv.min(), zv.max(), 12 if kind != "quadratic" else 20)
        X_mesh, Z_mesh = np.meshgrid(x_surf, z_surf)

        if kind == "linear":
            X_data = np.column_stack([xv, zv, np.ones(len(xv))])
            a, b, c = np.linalg.lstsq(X_data, yv, rcond=None)[0]
            Y_mesh = a * X_mesh + b * Z_mesh + c

        elif kind == "quadratic":
            X_data = np.column_stack([
                xv, zv, xv**2, zv**2, xv * zv, np.ones(len(xv))
            ])
            b1, b2, b3, b4, b5, b0 = np.linalg.lstsq(X_data, yv, rcond=None)[0]
            Y_mesh = (
                b0 + b1 * X_mesh + b2 * Z_mesh
                + b3 * X_mesh**2 + b4 * Z_mesh**2 + b5 * X_mesh * Z_mesh
            )

        else:  # logistic
            _require_binary_y(yv)
            X_data = sm.add_constant(np.column_stack([xv, zv]), has_constant="add")
            fitted = sm.Logit(yv, X_data).fit(method="lbfgs", maxiter=5000, disp=0)
            grid = sm.add_constant(
                np.column_stack([X_mesh.ravel(), Z_mesh.ravel()]),
                has_constant="add",
            )
            Y_mesh = fitted.predict(grid).reshape(X_mesh.shape)

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
