import matplotlib.pyplot as plt
from ..reg_modeling.reg import reg

def plot_residuals(df, y, x, dummies=None, logistic=False, dpi=150, figsize=(6, 4)):
    """
    Plot residuals vs fitted values to check linearity.
    
    Parameters: df, y, x, dummies, logistic
    Returns: None (displays plot)
    """
    results = reg(df, y, x, dummies=dummies, logistic=logistic, silent=True)
    
    if results is None:
        print("Error: Could not fit regression model")
        return
    
    residuals = results.resid
    fitted = results.fittedvalues
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(fitted, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
