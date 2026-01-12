def reg(df, y, x, dummies=None, logistic=False):
    """
    Run linear (OLS) or logistic regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables
    y : str
        Name of the dependent variable column
    x : str or list of str
        Name(s) of independent variable column(s)
    dummies : str or list of str, optional
        Categorical variable(s) to convert to dummy variables
    logistic : bool, default False
        If True, run logistic regression; otherwise run OLS
    
    Returns
    -------
    statsmodels results object or None if fitting fails
    """
    import statsmodels.api as sm
    import pandas as pd
    
    # Convert x to list if string, make copy to avoid modifying original
    if isinstance(x, str):
        x = [x]
    else:
        x = list(x)
    
    df_reg = df.copy()
    
    # Convert main variables to numeric
    for col in [y] + x:
        df_reg[col] = pd.to_numeric(df_reg[col], errors='coerce')
    
    # Handle dummy variables
    if dummies:
        if isinstance(dummies, str):
            dummies = [dummies]
        
        for dummy_var in dummies:
            # For logistic: filter to categories with variation in y
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
    
    # Drop rows with missing values
    df_reg = df_reg.dropna(subset=[y] + x)
    
    if len(df_reg) == 0:
        print("Error: No observations remaining after dropping missing values")
        return None
    
    # Prepare X and y
    X = sm.add_constant(df_reg[x].astype(float))
    y_data = df_reg[y].astype(float)
    
    # Fit model
    try:
        if logistic:
            model = sm.Logit(y_data, X)
            results = model.fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.OLS(y_data, X)
            results = model.fit()
        
        print(results.summary())
        return results
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        
        # For logistic with dummies, try again with larger categories only
        if logistic and dummies:
            min_obs = 30
            print(f"\nRetrying with categories having at least {min_obs} observations...")
            
            for dummy_var in dummies:
                value_counts = df_reg[dummy_var].value_counts()
                valid_categories = value_counts[value_counts >= min_obs].index
                df_reg = df_reg[df_reg[dummy_var].isin(valid_categories)]
            
            if len(df_reg) == 0:
                print("Error: No observations remaining after filtering")
                return None
            
            try:
                X = sm.add_constant(df_reg[x].astype(float))
                y_data = df_reg[y].astype(float)
                
                model = sm.Logit(y_data, X)
                results = model.fit(method='bfgs', maxiter=100, disp=0)
                print("\nResults with reduced sample:")
                print(results.summary())
                return results
            except Exception as e2:
                print(f"Error fitting reduced model: {e2}")
        
        return None
