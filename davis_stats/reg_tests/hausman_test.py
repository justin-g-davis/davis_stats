import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def hausman_test(df, y, x, entity, time, dummies=None, silent=False):
    """
    Hausman test: Fixed Effects (FE) vs Random Effects (RE).

    H0: RE is consistent and efficient (prefer RE)
    H1: RE is inconsistent (prefer FE)

    Parameters:
        df, y, x, entity, time, dummies, silent
    Returns:
        dict with test statistic, dof, p-value, and recommendation
        (or None if model/test cannot be computed)
    """
    # Normalize x input
    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)

    df_reg = df.copy()

    # Keep only needed columns
    needed_cols = [y, entity, time] + x
    if dummies:
        if isinstance(dummies, str):
            dummies = [dummies]
        needed_cols += dummies

    df_reg = df_reg[needed_cols].copy()

    # Numeric conversion for y/x
    for col in [y] + x:
        df_reg[col] = pd.to_numeric(df_reg[col], errors="coerce")

    # Handle optional dummies (same style as your reg function)
    if dummies:
        for dummy_var in dummies:
            dummy_cols = pd.get_dummies(
                df_reg[dummy_var],
                prefix=dummy_var,
                drop_first=True,
                dtype=float
            )
            df_reg = pd.concat([df_reg, dummy_cols], axis=1)
            x.extend(dummy_cols.columns.tolist())

    # Drop missing
    df_reg = df_reg.dropna(subset=[y, entity, time] + x)

    if len(df_reg) == 0:
        if not silent:
            print("Error: No observations remaining after dropping missing values")
        return None

    # Sort and set panel index
    df_reg = df_reg.sort_values([entity, time]).set_index([entity, time])

    y_data = df_reg[y].astype(float)
    X = df_reg[x].astype(float)

    # ---------- FE (within transformation) ----------
    grouped_y = y_data.groupby(level=0)
    grouped_X = X.groupby(level=0)

    y_fe = y_data - grouped_y.transform("mean")
    X_fe = X - grouped_X.transform("mean")

    # Drop columns with no within variation
    keep_cols = X_fe.columns[X_fe.var() > 0]
    X_fe = X_fe[keep_cols]
    X_re = X[keep_cols]

    if X_fe.shape[1] == 0:
        if not silent:
            print("Error: No regressors with within-entity variation for FE model.")
        return None

    # Align y with FE matrix
    valid_idx = X_fe.dropna().index
    X_fe = X_fe.loc[valid_idx]
    y_fe = y_fe.loc[valid_idx]
    X_re = X_re.loc[valid_idx]
    y_re = y_data.loc[valid_idx]

    try:
        fe_res = sm.OLS(y_fe, X_fe).fit()
        re_res = sm.MixedLM(y_re, sm.add_constant(X_re, has_constant='add'),
                            groups=pd.Index(valid_idx.get_level_values(0))).fit(reml=False, disp=False)
    except Exception as e:
        if not silent:
            print(f"Error fitting FE/RE models: {e}")
        return None

    # Coefs/cov only on common slope terms (exclude RE const)
    fe_beta = fe_res.params
    re_beta_all = re_res.params

    common = [c for c in fe_beta.index if c in re_beta_all.index]
    if len(common) == 0:
        if not silent:
            print("Error: No common coefficients between FE and RE models.")
        return None

    b_fe = fe_beta[common].values
    b_re = re_beta_all[common].values

    V_fe = fe_res.cov_params().loc[common, common].values
    V_re = re_res.cov_params().loc[common, common].values

    diff_b = b_fe - b_re
    diff_V = V_fe - V_re

    # Hausman statistic
    try:
        inv_diff_V = np.linalg.inv(diff_V)
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse if singular
        inv_diff_V = np.linalg.pinv(diff_V)

    h_stat = float(diff_b.T @ inv_diff_V @ diff_b)
    dof = len(common)
    p_value = float(1 - stats.chi2.cdf(h_stat, dof))

    result = {
        "test": "Hausman",
        "statistic": h_stat,
        "df": dof,
        "p_value": p_value,
        "decision": "Reject H0 (prefer FE)" if p_value <= 0.05 else "Fail to reject H0 (prefer RE)",
        "coefficients_tested": common
    }

    if not silent:
        print("Hausman Test (FE vs RE):\n")
        print("H0: Random Effects is consistent and efficient.")
        print("H1: Random Effects is inconsistent; Fixed Effects preferred.\n")
        print(f"Chi-square statistic: {h_stat:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_value:.4f}")
        print(f"Decision: {result['decision']}")

    return result
