import pandas as pd
import statsmodels.api as sm

def reg(df, y, x, dummies=None, logistic=False, robust=False, silent=False):
    """
    Run OLS or logistic regression.
    
    Parameters: df, y, x, dummies, logistic, robust, silent
    Returns: statsmodels results object
    """
    
    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)
    
    df_reg = df.copy()
    
    for col in [y] + x:
        df_reg[col] = pd.to_numeric(df_reg[col], errors='coerce')
    
    if dummies:
        if isinstance(dummies, str):
            dummies = [dummies]
        
        for dummy_var in dummies:
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
    
    df_reg = df_reg.dropna(subset=[y] + x)
    
    if len(df_reg) == 0:
        if not silent:
            print("Error: No observations remaining after dropping missing values")
        return None
    
    X = sm.add_constant(df_reg[x].astype(float))
    y_data = df_reg[y].astype(float)
    
    try:
        if logistic:
            model = sm.Logit(y_data, X)
            if robust:
                results = model.fit(method='lbfgs', maxiter=5000, disp=0, cov_type='HC1')
            else:
                results = model.fit(method='lbfgs', maxiter=5000, disp=0)
        else:
            model = sm.OLS(y_data, X)
            if robust:
                results = model.fit(cov_type='HC1')
            else:
                results = model.fit()
        
        if not silent:
            print(results.summary())
        return results
        
    except Exception as e:
        if not silent:
            print(f"Error fitting model: {e}")
        return None
