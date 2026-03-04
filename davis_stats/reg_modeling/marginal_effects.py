import pandas as pd
import statsmodels.api as sm

def marginal_effects(df, y, x, dummies=None, logistic=False, robust=False):
    """
    Run regression and display marginal effects instead of coefficients.
    
    Parameters: df, y, x, dummies, logistic, robust
    Returns: tuple (model_results, marginal_effects_results)
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
        print("Error: No observations remaining after dropping missing values")
        return None, None
    
    X = sm.add_constant(df_reg[x].astype(float))
    y_data = df_reg[y].astype(float)
    
    try:
        if logistic:
            model = sm.Logit(y_data, X)
            if robust:
                results = model.fit(method='lbfgs', maxiter=5000, disp=0, cov_type='HC1')
            else:
                results = model.fit(method='lbfgs', maxiter=5000, disp=0)
            
            # Calculate marginal effects (always at mean)
            try:
                me = results.get_margeff(at='mean')
                
                print("="*70)
                print(f"MARGINAL EFFECTS - {y.upper()}")
                print("="*70)
                print("Marginal effects evaluated at the mean of all variables")
                print()
                print(me.summary())
                
                return results, me
                
            except Exception as me_error:
                print(f"Error calculating marginal effects: {me_error}")
                print("\nShowing regression coefficients instead:")
                print(results.summary())
                return results, None
                
        else:
            # For OLS, marginal effects = coefficients
            model = sm.OLS(y_data, X)
            if robust:
                results = model.fit(cov_type='HC1')
            else:
                results = model.fit()
            
            print("="*70)
            print(f"MARGINAL EFFECTS - {y.upper()} (OLS)")
            print("="*70)
            print("Note: For OLS, marginal effects equal coefficients")
            print()
            print(results.summary())
            
            return results, None
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, None
