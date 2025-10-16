import math
import warnings
import pandas as pd

def try_statsmodels_conditional_logit(design: pd.DataFrame, formula_cols, y_col="is_used", strata_col="stratum_id"):
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit
        y = design[y_col].values
        X = design[list(formula_cols)]
        groups = design[strata_col].values
        model = ConditionalLogit(y, X, groups=groups)
        res = model.fit(disp=False)
        return res
    except Exception as e:
        warnings.warn(f"ConditionalLogit unavailable or failed: {e}")
        return None

def export_for_r(design: pd.DataFrame, formula_cols, out_csv="issa_design.csv"):
    cols = ["stratum_id","is_used"] + list(formula_cols)
    design[cols].to_csv(out_csv, index=False)
    msg = f"Exported {out_csv}. In R: survival::clogit(is_used ~ {' + '.join(formula_cols)} + strata(stratum_id), data=read.csv('{out_csv}'))"
    print(msg)
    return out_csv, msg

def summarize_conditional_logit(res):
    """Return a dict with metrics (AIC, BIC, logLik, df) if available."""
    try:
        ll = float(res.llf)
        k = int(res.df_model + 1)  # +1 for intercept if present; if not, leave df_model
        n = int(res.nobs)
        aic = 2*k - 2*ll
        bic = math.log(n)*k - 2*ll
        return {"ll": ll, "k": k, "n": n, "AIC": aic, "BIC": bic}
    except Exception:
        return {}
