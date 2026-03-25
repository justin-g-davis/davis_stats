import numpy as np
import pandas as pd
from scipy import stats
from ..reg_modeling.reg import reg

import numpy as np
from scipy import stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from ..reg_modeling.reg import reg

def hausman_test(df, y, x, dummies=None, entity=None, time=None, robust=False):
    """
    Hausman test for panel models: FE vs RE.
    """

    robust = False  # force off for Hausman consistency

    if entity is None:
        print("Error: entity is required for Hausman test.")
        return None

    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)

    fe_res = reg(df=df, y=y, x=x, dummies=dummies, logistic=False, panel="fe", entity=entity, time=time, robust=False, silent=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        re_res = reg(df=df, y=y, x=x, dummies=dummies, logistic=False, panel="re", entity=entity, time=time, robust=False, silent=True)
        re_warnings = [str(msg.message) for msg in w if issubclass(msg.category, ConvergenceWarning)]

    if fe_res is None or re_res is None:
        print("Error: Could not fit FE and/or RE model for Hausman test.")
        return None

    if hasattr(re_res, "converged") and not re_res.converged:
        print("Error: RE model did not converge. Hausman test is not valid.")
        return None

    try:
        fe_names = list(fe_res.params.index)
        re_names = list(re_res.params.index)
        common = [n for n in fe_names if n in re_names and n != "const"]

        if len(common) == 0:
            print("Error: No common slope coefficients between FE and RE.")
            return None

        b_fe = fe_res.params.loc[common].to_numpy(dtype=float)
        b_re = re_res.params.loc[common].to_numpy(dtype=float)

        V_fe = fe_res.cov_params().loc[common, common].to_numpy(dtype=float)
        V_re = re_res.cov_params().loc[common, common].to_numpy(dtype=float)

        diff_b = b_fe - b_re
        diff_V = 0.5 * ((V_fe - V_re) + (V_fe - V_re).T)

        eigvals = np.linalg.eigvalsh(diff_V)
        if np.min(eigvals) < -1e-8:
            print("Error: Var(FE)-Var(RE) is not positive semidefinite. Hausman test invalid.")
            return None

        if (not np.isfinite(np.linalg.cond(diff_V))) or (np.linalg.cond(diff_V) > 1e12):
            print("Error: Var(FE)-Var(RE) is near-singular/ill-conditioned. Hausman test invalid.")
            return None

        h_stat = float(diff_b.T @ np.linalg.inv(diff_V) @ diff_b)
        if h_stat < -1e-8:
            print("Error: Negative Hausman statistic due to numerical instability. Test invalid.")
            return None

        h_stat = max(h_stat, 0.0)
        dof = len(common)
        p_value = float(1 - stats.chi2.cdf(h_stat, dof))
        decision = "Reject H0 (prefer FE): RE likely inconsistent." if p_value <= 0.05 else "Fail to reject H0 (prefer RE): RE appears consistent."

        result = {
            "test": "Hausman",
            "statistic": h_stat,
            "df": dof,
            "p_value": p_value,
            "decision": decision,
            "coefficients_tested": common,
            "re_convergence_warnings": re_warnings,
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
        if re_warnings:
            print("\nNote: RE model emitted convergence warnings; interpret with caution.")

        return result

    except Exception as e:
        print(f"Error computing Hausman test: {e}")
        return None
