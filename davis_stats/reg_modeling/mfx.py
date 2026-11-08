def mfx(results, at="overall", silent=False):
    """
    Marginal effects from a fitted statsmodels result.

    Parameters
    ----------
    results : fitted model from reg(...)
    at : {'overall', 'mean', 'median', 'zero'}
        'overall' = average marginal effects (AME)
        'mean'    = marginal effects at the mean (MEM)
    silent : if True, do not print

    Returns
    -------
    MarginalEffectsResults for Logit; results itself for OLS/linear models;
    None on failure.
    """
    if results is None:
        if not silent:
            print("Error: results is None.")
        return None

    # Linear models: coefficients already are marginal effects
    if not hasattr(results, "get_margeff"):
        if not silent:
            print("=" * 70)
            print("MARGINAL EFFECTS (linear model)")
            print("=" * 70)
            print("For OLS / linear models, marginal effects equal coefficients.")
            print()
            print(results.summary())
        return results

    try:
        me = results.get_margeff(at=at)

        if not silent:
            label = {
                "overall": "Average marginal effects (AME)",
                "mean": "Marginal effects at the mean (MEM)",
                "median": "Marginal effects at the median",
                "zero": "Marginal effects at zero",
            }.get(at, f"Marginal effects at '{at}'")

            print("=" * 70)
            print("MARGINAL EFFECTS")
            print("=" * 70)
            print(label)
            print()
            print(me.summary())

        return me

    except Exception as e:
        if not silent:
            print(f"Error calculating marginal effects: {e}")
        return None
