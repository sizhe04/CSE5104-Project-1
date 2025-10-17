from typing import Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm


def ols_pvalues(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_const = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_const)
    res = model.fit()
    pvals = res.pvalues.drop(labels=["const"], errors="ignore")
    out = pd.DataFrame({"feature": pvals.index, "p_value": pvals.values})
    out.sort_values("p_value", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out





