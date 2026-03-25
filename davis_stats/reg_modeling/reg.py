import pandas as pd
import statsmodels.api as sm

def reg(
    df,
    y,
    x,
    dummies=None,
    logistic=False,      # keeps old behavior
    panel=None,          # None | "fe" | "re"
    entity=None,         # required when panel in {"fe","re"}
    time=None,           # optional panel sort key
    robust=False,
    silent=False
):
    """
    Run pooled OLS/logit, or panel FE/RE.

    Parameters:
        df, y, x, dummies, logistic, panel, entity, time, robust, silent
    Returns:
        statsmodels results object (or None)
    """

    # guardrails
    if panel not in (None, "fe", "re"):
        if not silent:
            print("Error: panel must be None, 'fe', or 're'.")
        return None

    if logistic and panel is not None:
        if not silent:
            print("Error: logistic with panel FE/RE is not supported in this reg function.")
        return None

    # normalize x
    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)

    df_reg = df.copy()

    # numeric conversion
    for col in [y] + x:
        df_reg[col] = pd.to_numeric(df_reg[col], errors="coerce")

    # dummies
    if dummies:
        if isinstance(dummies, str):
            dummies = [dummies]

        for dummy_var in dummies:
            # Keep your existing logistic category filter behavior
            if logistic:
                ct = pd.crosstab(df_reg[dummy_var], df_reg[y])
                valid_categories = ct[(ct > 0).all(axis=1)].index
                df_reg = df_reg[df_reg[dummy_var].isin(valid_categories)]

            dummy_cols = pd.get_dummies(
                df_reg[dummy_var],
                prefix=dummy_var,
                drop_first=True,
                dtype=float
            )
            df_reg = pd.concat([df_reg, dummy_cols], axis=1)
            x.extend(dummy_cols.columns.tolist())

    # panel regression
    if panel in ("fe", "re"):
        if entity is None:
            if not silent:
                print("Error: entity is required for panel='fe' or panel='re'.")
            return None

        required = [y] + x + [entity] + ([time] if time else [])
        df_reg = df_reg.dropna(subset=required)

        if len(df_reg) == 0:
            if not silent:
                print("Error: No observations remaining after dropping missing values")
            return None

        sort_cols = [entity] + ([time] if time else [])
        df_reg = df_reg.sort_values(sort_cols)

        y_data = df_reg[y].astype(float)
        X_data = df_reg[x].astype(float)

        try:
            if panel == "fe":
                # Within transformation by entity
                y_fe = y_data - y_data.groupby(df_reg[entity]).transform("mean")
                X_fe = X_data - X_data.groupby(df_reg[entity]).transform("mean")

                # Remove regressors with no within variation
                keep_cols = X_fe.columns[X_fe.var() > 0]
                X_fe = X_fe[keep_cols]

                if X_fe.shape[1] == 0:
                    if not silent:
                        print("Error: No regressors with within-entity variation for FE.")
                    return None

                model = sm.OLS(y_fe, X_fe)
                results = model.fit(cov_type="HC1") if robust else model.fit()

            else:  # panel == "re"
                X_re = sm.add_constant(X_data, has_constant="add")
                model = sm.MixedLM(endog=y_data, exog=X_re, groups=df_reg[entity])
                # MixedLM robust covariance is not the same as OLS/Logit HC1
                if robust and not silent:
                    print("Note: robust=True is not applied for MixedLM RE in this implementation.")
                results = model.fit(reml=False, disp=False)

            if not silent:
                print(results.summary())
            return results

        except Exception as e:
            if not silent:
                print(f"Error fitting model: {e}")
            return None

    # pooled OLS or logistic regression
    df_reg = df_reg.dropna(subset=[y] + x)

    if len(df_reg) == 0:
        if not silent:
            print("Error: No observations remaining after dropping missing values")
        return None

    X = sm.add_constant(df_reg[x].astype(float), has_constant="add")
    y_data = df_reg[y].astype(float)

    try:
        if logistic:
            model = sm.Logit(y_data, X)
            if robust:
                results = model.fit(method="lbfgs", maxiter=5000, disp=0, cov_type="HC1")
            else:
                results = model.fit(method="lbfgs", maxiter=5000, disp=0)
        else:
            model = sm.OLS(y_data, X)
            if robust:
                results = model.fit(cov_type="HC1")
            else:
                results = model.fit()

        if not silent:
            print(results.summary())
        return results

    except Exception as e:
        if not silent:
            print(f"Error fitting model: {e}")
        return None
