import numpy as np, pandas as pd
from collections import defaultdict

def softmax_by_stratum(lp: pd.Series, strata: pd.Series) -> pd.Series:
    df = pd.DataFrame({"lp": lp, "s": strata})
    df["exp"] = np.exp(df["lp"] - df.groupby("s")["lp"].transform("max"))
    df["sumexp"] = df.groupby("s")["exp"].transform("sum")
    return df["exp"] / df["sumexp"]

def kfold_time_cv(design: pd.DataFrame, cols: list, k: int = 5):
    """Return CV metrics dict using statsmodels ConditionalLogit if available."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    metrics = defaultdict(list)
    # build fold labels per id by time order
    d = design.copy()
    d = d.sort_values(["id","time"])
    d["t_rank"] = d.groupby("id")["time"].rank(method="first")
    d["t_fold"] = (d["t_rank"] % k).astype(int)
    for f in range(k):
        train = d[d["t_fold"] != f]
        test  = d[d["t_fold"] == f]
        y_tr, X_tr, g_tr = train["is_used"].values, train[cols], train["stratum_id"].values
        try:
            res = ConditionalLogit(y_tr, X_tr, groups=g_tr).fit(disp=False)
            # predict on test
            eta = (test[cols] @ res.params).astype(float)
            p = softmax_by_stratum(eta, test["stratum_id"])
            # collect metrics
            used_mask = test["is_used"].values == 1
            p_used = p[used_mask]
            metrics["log_score"].append(float((-np.log(p_used)).mean()))
            # ranking: rank 1 = best
            test2 = test.assign(lp=eta, p=p)
            ranks = test2.groupby("stratum_id")["lp"].rank(ascending=False, method="min")
            rank_used = ranks[used_mask]
            metrics["rank_mean"].append(float(rank_used.mean()))
            metrics["rank_top1"].append(float((rank_used==1).mean()))
        except Exception as e:
            metrics["errors"].append(str(e))
            break
    # aggregate
    out = {k: float(np.mean(v)) for k,v in metrics.items() if k not in ("errors",)}
    if metrics["errors"]:
        out["errors"] = metrics["errors"]
    return out
