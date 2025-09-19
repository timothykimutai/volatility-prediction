import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict

from .performance import calculate_portfolio_performance


def _mvo_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Objective function for MVO: minimize portfolio variance."""
    return float(weights.T @ cov_matrix @ weights)


def _max_sharpe_objective(weights: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """Objective function for maximizing Sharpe Ratio (by minimizing its negative)."""
    portfolio_return = float(weights.T @ mu)
    portfolio_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    if portfolio_vol <= 0:
        return float('inf')
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    return -float(sharpe_ratio)


def calculate_mvo_weights(
    mu: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float
) -> Tuple[pd.Series, Dict[str, float]]:
    num_assets = len(mu)
    args = (mu.values, cov_matrix.values, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1)},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    result = minimize(
        _max_sharpe_objective,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError("MVO optimization failed: " + result.message)

    weights = pd.Series(result.x, index=mu.index)
    performance = calculate_portfolio_performance(weights, mu, cov_matrix, risk_free_rate)

    return weights, performance


def calculate_efficient_frontier(
    mu: pd.Series, cov_matrix: pd.DataFrame, num_points: int = 50
) -> pd.DataFrame:
    num_assets = len(mu)
    target_returns = np.linspace(float(mu.min()), float(mu.max()), num_points)
    frontier_vols = []

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1)},
            {'type': 'eq', 'fun': lambda w: float(w.T @ mu.values - target_return)}
        )
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        result = minimize(
            _mvo_objective,
            initial_weights,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            frontier_vols.append(float(np.sqrt(result.fun)))

    return pd.DataFrame({'Return': target_returns[:len(frontier_vols)], 'Volatility': frontier_vols})