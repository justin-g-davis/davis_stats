import numpy as np
import pandas as pd
from scipy import stats
from ..reg_modeling.reg import reg

def hausman_test(df, y, x, dummies=None, entity=None, time=None, robust=False):
    """
    Hausman test for panel models: FE vs RE.

    H0: RE is consistent and efficient (prefer RE)
    H1: RE is inconsistent (prefer FE)

    Parameters:
        df, y, x, dummies, entity, time, robust, silent
    Returns:
        dict with Hausman test results (or None on failure)
    """

    if entity is None:
        print("Error: entity is required for Hausman test.")
        return None

    # Normalize x for local checks
    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)

    # Fit FE and RE using the same reg interface
    fe_res = reg(df=df, y=y, x=x, dummies=dummies, logistic=False, panel="fe", entity=entity, time=time, robust=robust, silent=True)

    re_res = reg(df=df, y=y, x=x, dummies=dummies, logistic=False, panel="re", entity=entity, time=time, robust=robust, silent=True)

    if fe_res is None or re_res is None:
        print("Error: Could not fit FE and/or RE model for Hausman test.")
        return None

    try:
        # FE has no constant; RE usually has const in MixedLM
        fe_names = list(fe_res.params.index)
        re_names = list(re_res.params.index)

        common = [name for name in fe_names if name in re_names and name != "const"]

        if len(common) == 0:
            print("Error: No common slope coefficients between FE and RE.")
            return None

        b_fe = fe_res.params.loc[common].values
        b_re = re_res.params.loc[common].values

        V_fe = fe_res.cov_params().loc[common, common].values
        V_re = re_res.cov_params().loc[common, common].values

        diff_b = b_fe - b_re
        diff_V = V_fe - V_re

        # Invert variance difference (fallback to pseudo-inverse if singular)
        try:
            inv_diff_V = np.linalg.inv(diff_V)
        except np.linalg.LinAlgError:
            inv_diff_V = np.linalg.pinv(diff_V)

        h_stat = float(diff_b.T @ inv_diff_V @ diff_b)
        dof = len(common)
        p_value = float(1 - stats.chi2.cdf(h_stat, dof))

        decision = (
            "Reject H0 (prefer FE): RE likely inconsistent."
            if p_value <= 0.05
            else "Fail to reject H0 (prefer RE): RE appears consistent."
        )

        result = {"test": "Hausman", "statistic": h_stat, "df": dof, "p_value": p_value, "decision": decision, "coefficients_tested": common}

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
