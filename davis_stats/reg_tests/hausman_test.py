import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects

def hausman_test(df, y, x, dummies=None, entity=None, time=None, robust=False):
    """
    Hausman test for panel models: FE vs RE.

    H0: RE is consistent and efficient (prefer RE)
    H1: RE is inconsistent (prefer FE)

    Returns dict with fields: test, statistic, df, p_value, decision, coefficients_tested
    """

    if entity is None or time is None:
        print("Error: entity and time are required for Hausman test.")
        return None

    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)

    df_reg = df.copy()

    # numeric conversion for y/x
    for col in [y] + x:
        df_reg[col] = pd.to_numeric(df_reg[col], errors="coerce")

    # optional dummies (match your reg style)
    if dummies:
        if isinstance(dummies, str):
            dummies = [dummies]
        for dummy_var in dummies:
            dummy_cols = pd.get_dummies(df_reg[dummy_var], prefix=dummy_var, drop_first=True, dtype=float)
            df_reg = pd.concat([df_reg, dummy_cols], axis=1)
            x.extend(dummy_cols.columns.tolist())

    df_reg = df_reg.dropna(subset=[y, entity, time] + x)
    if len(df_reg) == 0:
        print("Error: No observations remaining after dropping missing values")
        return None

    # panel index for linearmodels
    df_reg = df_reg.sort_values([entity, time]).set_index([entity, time])

    y_data = df_reg[y].astype(float)
    X_data = df_reg[x].astype(float)

    try:
        # FE (entity effects)
        fe_model = PanelOLS(y_data, X_data, entity_effects=True, drop_absorbed=True)
        fe_res = fe_model.fit(cov_type="unadjusted")

        # RE
        re_model = RandomEffects(y_data, X_data)
        re_res = re_model.fit(cov_type="unadjusted")

        fe_names = list(fe_res.params.index)
        re_names = list(re_res.params.index)
        common = [name for name in fe_names if name in re_names]

        if len(common) == 0:
            print("Error: No common slope coefficients between FE and RE.")
            return None

        b_fe = fe_res.params.loc[common].to_numpy(dtype=float)
        b_re = re_res.params.loc[common].to_numpy(dtype=float)

        V_fe = fe_res.cov.loc[common, common].to_numpy(dtype=float)
        V_re = re_res.cov.loc[common, common].to_numpy(dtype=float)

        diff_b = b_fe - b_re
        diff_V = V_fe - V_re
        diff_V = 0.5 * (diff_V + diff_V.T)  # symmetrize

        # Generalized (stable) Hausman with rank-aware dof and ridge fallback
        eigvals = np.linalg.eigvalsh(diff_V)
        tol = 1e-8
        rank = int(np.sum(eigvals > tol))

        if rank == 0:
            lam = 1e-8 * max(1.0, float(np.mean(np.diag(V_fe + V_re))))
            diff_V = diff_V + lam * np.eye(diff_V.shape[0])
            eigvals = np.linalg.eigvalsh(diff_V)
            rank = int(np.sum(eigvals > tol))

        if rank == 0:
            h_stat = 0.0
            dof = len(common)
        else:
            inv_diff_V = np.linalg.pinv(diff_V, rcond=1e-10)
            h_stat = float(diff_b.T @ inv_diff_V @ diff_b)
            h_stat = max(h_stat, 0.0)
            dof = rank

        p_value = float(1 - stats.chi2.cdf(h_stat, dof))
        decision = (
            "Reject H0 (prefer FE): RE likely inconsistent."
            if p_value <= 0.05
            else "Fail to reject H0 (prefer RE): RE appears consistent."
        )

        result = {
            "test": "Hausman",
            "statistic": h_stat,
            "df": dof,
            "p_value": p_value,
            "decision": decision,
            "coefficients_tested": common
        }

        print("Hausman test (FE vs RE):\n")
        print("H0: RE is consistent and efficient (prefer RE).")
        print("H1: RE is inconsistent (prefer FE).\n")
        print("If p-value ≤ 0.05: Reject H0, prefer FE.")
        print("If p-value > 0.05: Fail to reject H0, prefer RE.\n")
        print(f"Chi-square statistic: {h_stat:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_value:.4f}")
        print(f"Decision: {decision}")

        return result

    except Exception as e:
        print(f"Error computing Hausman test: {e}")
        return None
