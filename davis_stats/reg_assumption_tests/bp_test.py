from statsmodels.stats.diagnostic import het_breuschpagan
from ..reg_modeling.reg import reg

def bp_test(df, y, x, dummies=None, logistic=False):
    """
    Breusch-Pagan test for constant error variance (homoscedasticity).
    """
    print("Breusch-Pagan test for homoscedasticity:\n")
    
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    
    if results is None:
        print("Error: Could not fit regression model.\n")
        return None
    
    if logistic:
        print("Note: Breusch-Pagan test is for OLS only.\n")
        return None
    
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(results.resid, results.model.exog)
    print("If p-value ≤ 0.5: Reject H0, assume heteroscedasticity. Remedies needed.")
    print("If p-value > 0.5: Fail to reject H0, assume homoscedasticity.\n")
    print(f"p-value: {lm_pvalue:.4f}")
