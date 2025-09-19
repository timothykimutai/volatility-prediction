import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Any


def _risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    weights = np.array(weights)
    portfolio_variance = float(weights.T @ cov_matrix @ weights)

    # Risk contributions
    marginal_contrib = (cov_matrix @ weights)
    risk_contrib = weights * marginal_contrib

    # Objective: sum of squared differences of risk contributions from the mean contribution
    target_contrib = portfolio_variance / float(len(weights))
    return float(np.sum((risk_contrib - target_contrib) ** 2))


def calculate_risk_parity_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    num_assets = cov_matrix.shape[0]
    initial_weights = np.ones(num_assets) / num_assets

    constraints = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1)},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    result = minimize(
        _risk_parity_objective,
        initial_weights,
        args=(cov_matrix.values,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12}
    )

    if not result.success:
        raise ValueError("Risk-Parity optimization failed: " + result.message)

    return pd.Series(result.x, index=cov_matrix.columns)