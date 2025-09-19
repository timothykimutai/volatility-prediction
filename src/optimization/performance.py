import numpy as np
import pandas as pd
from typing import Dict


def calculate_portfolio_performance(
    weights: pd.Series, mu: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float
) -> Dict[str, float]:
    expected_return = float(np.sum(weights * mu))
    w = weights.values if hasattr(weights, 'values') else np.array(weights)
    sigma = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
    volatility = float(np.sqrt(np.dot(w.T, np.dot(sigma, w))))
    sharpe_ratio = float((expected_return - risk_free_rate) / volatility) if volatility > 0 else 0.0
    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }


def calculate_risk_contribution(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    w = weights.values
    sigma = cov_matrix.values
    portfolio_vol = float(np.sqrt(w.T @ sigma @ w))

    if portfolio_vol < 1e-8:
        return pd.Series(0.0, index=weights.index)

    # Marginal Risk Contribution (MRC)
    mrc = (sigma @ w) / portfolio_vol
    # Risk Contribution (RC)
    rc = w * mrc
    # Percentage Risk Contribution
    prc = rc / float(np.sum(rc))

    return pd.Series(prc, index=weights.index)