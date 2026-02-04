from statsmodels.stats.diagnostic import het_breuschpagan
from .reg import reg

def bp_test(df, y, x, dummies=None, logistic=False, alpha=0.05):
    """
    Breusch-Pagan test for constant error variance (homoscedasticity).
    """
    print("Running Breusch-Pagan test for homoscedasticity...\n")
    
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    
    if results is None:
        print("Error: Could not fit regression model.\n")
        return None
    
    if logistic:
        print("Note: Breusch-Pagan test is for OLS only.\n")
        return None
    
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(results.resid, results.model.exog)
    
    print("\n" + "="*70)
    print("BREUSCH-PAGAN TEST FOR HOMOSCEDASTICITY")
    print("="*70)
    print(f"LM statistic: {lm_stat:.4f}")
    print(f"p-value: {lm_pvalue:.4f}")
    
    if lm_pvalue < alpha:
        print(f"\n✗ REJECT H0 (p < {alpha}): Heteroscedasticity detected")
        print("\nREMEDIES:")
        print("  1. Use robust standard errors: reg(..., robust=True)")
        print("  2. Transform dependent variable (log, square root)")
        print("  3. Use weighted least squares")
    else:
        print(f"\n✓ FAIL TO REJECT H0 (p >= {alpha}): Homoscedasticity")
    
    print("="*70)
    
    return {
        'lm_statistic': lm_stat,
        'lm_pvalue': lm_pvalue,
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        'heteroscedasticity_detected': lm_pvalue < alpha}
