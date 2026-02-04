import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ..reg_modeling.reg import reg

def vif_test(df, y, x, dummies=None, logistic=False):
    """
    Calculate VIF to detect multicollinearity.
    """
    print("Variance Inflation Factors (VIF):\n")
    
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    
    if results is None:
        print("Error: Could not fit regression model.\n")
        return None
    
    X = results.model.exog
    var_names = results.model.exog_names
    
    if 'const' in var_names:
        const_idx = var_names.index('const')
        X = np.delete(X, const_idx, axis=1)
        var_names = [v for v in var_names if v != 'const']
    
    if X.shape[1] == 0:
        print("Error: No independent variables.\n")
        return None
    
    if X.shape[1] == 1:
        print("Only one variable. VIF not applicable.\n")
        return None
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = var_names
    
    try:
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    except Exception as e:
        print(f"Error calculating VIF: {e}\n")
        return None
    
    print("VIF < 5: No multicollinearity")
    print("VIF 5-10: Moderate multicollinearity")
    print("VIF > 10: Severe multicollinearity. Further remedies needed.\n")
    print(vif_data.to_string(index=False))
