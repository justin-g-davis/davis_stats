from statsmodels.stats.stattools import durbin_watson
from ..reg_modeling.reg import reg

def dw_test(df, y, x, dummies=None, logistic=False):
    """
    Durbin-Watson test for autocorrelation.
    """
    print("Durbin-Watson test for autocorrelation:\n")
    
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    
    if results is None:
        print("Error: Could not fit regression model.\n")
        return None
    
    if logistic:
        print("Note: Durbin-Watson test is for OLS only.\n")
        return None
    
    dw_stat = durbin_watson(results.resid)
    
    print("DW ≈ 2: No autocorrelation")
    print("DW 1.5-2.5: Generally acceptable.")
    print("DW < 1.5: Positive autocorrelation. Remedies needed.")
    print("DW > 2.5: Negative autocorrelation. Remedies needed.\n")
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
