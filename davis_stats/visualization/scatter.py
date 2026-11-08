import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def scatter(df, y, x, fit_line=False, color_scale=False, title=None, figsize=(7, 4.5)):
    if isinstance(x, str):
        x = [x]

    cols = [y] + x
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    yv = data[y].to_numpy(dtype=float)

    # --- 2D ---
    if len(x) == 1:
        xv = data[x[0]].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=figsize)

        if color_scale:
            sc = ax.scatter(xv, yv, c=yv, cmap="coolwarm", alpha=0.75)
            fig.colorbar(sc, ax=ax, label=y)
        else:
            ax.scatter(xv, yv, alpha=0.75)

        if fit_line == "linear":
            coefs = np.polyfit(xv, yv, 1)
            xs = np.linspace(xv.min(), xv.max(), 200)
            ax.plot(xs, np.polyval(coefs, xs), color="black")

        elif fit_line == "quadratic":
            coefs = np.polyfit(xv, yv, 2)
            xs = np.linspace(xv.min(), xv.max(), 200)
            ax.plot(xs, np.polyval(coefs, xs), color="black")

        elif fit_line == "logistic":
            fit = sm.Logit(yv, sm.add_constant(xv)).fit(disp=0)
            xs = np.linspace(xv.min(), xv.max(), 200)
            ax.plot(xs, fit.predict(sm.add_constant(xs)), color="black")

        elif fit_line:
            raise ValueError("fit_line must be False, 'linear', 'quadratic', or 'logistic'")

        ax.set_xlabel(x[0])
        ax.set_ylabel(y)
        ax.set_title(title or f"{y} and {x[0]}")
        plt.show(block=False)
        return ax

    # --- 3D ---
    xv = data[x[0]].to_numpy(dtype=float)
    zv = data[x[1]].to_numpy(dtype=float)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if color_scale:
        sc = ax.scatter(xv, zv, yv, c=yv, cmap="coolwarm", alpha=0.75)
        fig.colorbar(sc, ax=ax, shrink=0.7, label=y)
    else:
        ax.scatter(xv, zv, yv, alpha=0.75)

    xs = np.linspace(xv.min(), xv.max(), 20)
    zs = np.linspace(zv.min(), zv.max(), 20)
    Xg, Zg = np.meshgrid(xs, zs)

    if fit_line == "linear":
        X = np.column_stack([xv, zv, np.ones(len(xv))])
        a, b, c = np.linalg.lstsq(X, yv, rcond=None)[0]
        Yg = a * Xg + b * Zg + c
        ax.plot_surface(Xg, Zg, Yg, color="gray", alpha=0.3)

    elif fit_line == "quadratic":
        X = np.column_stack([
            xv, zv, xv**2, zv**2, xv * zv, np.ones(len(xv))
        ])
        b1, b2, b3, b4, b5, b0 = np.linalg.lstsq(X, yv, rcond=None)[0]
        Yg = b0 + b1 * Xg + b2 * Zg + b3 * Xg**2 + b4 * Zg**2 + b5 * Xg * Zg
        ax.plot_surface(Xg, Zg, Yg, color="gray", alpha=0.3)

    elif fit_line == "logistic":
        fit = sm.Logit(yv, sm.add_constant(np.column_stack([xv, zv]))).fit(disp=0)
        grid = sm.add_constant(np.column_stack([Xg.ravel(), Zg.ravel()]))
        Yg = fit.predict(grid).reshape(Xg.shape)
        ax.plot_surface(Xg, Zg, Yg, color="gray", alpha=0.3)

    elif fit_line:
        raise ValueError("fit_line must be False, 'linear', 'quadratic', or 'logistic'")

    ax.set_xlabel(x[0])
    ax.set_ylabel(x[1])
    ax.set_zlabel(y)
    ax.set_title(title or f"{y}, {x[0]}, and {x[1]}")
    plt.show(block=False)
    return ax
