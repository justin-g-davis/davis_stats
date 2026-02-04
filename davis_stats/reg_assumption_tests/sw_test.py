from scipy import stats
from .reg import reg

def sw_test(df, y, x, dummies=None, logistic=False):
    """
    Shapiro-Wilk test for normally distributed residuals.
    """
    print("Shapiro-Wilk test for normality:\n")
    
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    
    if results is None:
        print("Error: Could not fit regression model.\n")
        return None
    
    residuals = results.resid
    n = len(residuals)
    
    sw_stat, sw_pvalue = stats.shapiro(residuals)
    
    print("If p-value ≤ 0.05: Reject H0, assume non-normality. Further remedies may be needed but less critical with a large sample size.")
    print("If p-value > 0.05: Fail to reject H0, assume normality.\n")
    print(f"Sample size: {n}")
    print(f"p-value: {sw_pvalue:.4f}")
