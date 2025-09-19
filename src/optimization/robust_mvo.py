import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict

from .performance import calculate_portfolio_performance


def _enhanced_mvo_objective(
    weights: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    epsilon: np.ndarray,
    w_ref: np.ndarray,
    rho_rp: float,
    tau_l2: float,
    lambda_risk_aversion: float
) -> float:
    """
    The objective function for the enhanced robust MVO model.
    """
    # 1. Risk term (variance)
    risk_term = 0.5 * (weights.T @ cov_matrix @ weights)
    
    # 2. Robust return term
    # For long-only, worst-case mu is mu - epsilon
    # The term to maximize is w.T @ (mu - epsilon) = w.T @ mu - w.T @ epsilon
    # Since we minimize, the term is -lambda * (w.T @ mu - w.T @ epsilon)
    # Using L1 norm: w.T @ epsilon is equivalent to sum(epsilon_i * |w_i|)
    robust_return_term = -lambda_risk_aversion * (weights.T @ mu - np.sum(epsilon * np.abs(weights)))

    # 3. Risk-Parity regularization term
    rp_reg_term = 0.5 * rho_rp * np.sum((weights - w_ref)**2)

    # 4. L2 regularization term
    l2_reg_term = 0.5 * tau_l2 * np.sum(weights**2)
    
    return risk_term + robust_return_term + rp_reg_term + l2_reg_term

def calculate_enhanced_mvo_weights(
    mu: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    w_ref: pd.Series,
    epsilon: pd.Series,
    rho_rp: float,
    tau_l2: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Calculates portfolio weights using the enhanced robust MVO model.
    The risk aversion parameter lambda is implicitly derived by targeting max Sharpe.
    However, a direct max-Sharpe formulation with these penalties is non-trivial.
    We instead use a standard quadratic utility formulation where lambda acts as
    a proxy for risk aversion. A reasonable default can be the market Sharpe ratio.
    Here we set lambda to achieve a balance, a common value is derived from MVO.
    """
    num_assets = len(mu)
    
    # Estimate a reasonable risk aversion parameter from a simple MVO
    # A common proxy: lambda = (market_return - rf) / market_variance
    # Here, we use the expected return and variance of an equal-weight portfolio
    w_ew = np.ones(num_assets) / num_assets
    ew_ret = w_ew.T @ mu.values
    ew_var = w_ew.T @ cov_matrix.values @ w_ew
    lambda_risk_aversion = (ew_ret - risk_free_rate) / ew_var if ew_var > 0 else 1.0
    lambda_risk_aversion = max(0.1, lambda_risk_aversion) # Ensure it's positive
    
    args = (
        mu.values,
        cov_matrix.values,
        epsilon.values,
        w_ref.values,
        rho_rp,
        tau_l2,
        lambda_risk_aversion,
    )

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = w_ref.values # Start optimization from the RP reference

    result = minimize(
        _enhanced_mvo_objective,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError("Enhanced MVO optimization failed: " + result.message)
        
    weights = pd.Series(result.x, index=mu.index)
    performance = calculate_portfolio_performance(weights, mu, cov_matrix, risk_free_rate)
    
    return weights, performance