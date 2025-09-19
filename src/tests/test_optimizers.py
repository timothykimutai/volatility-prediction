import pandas as pd
import numpy as np
import pytest

from src.optimization.mvo import calculate_mvo_weights
from src.optimization.risk_parity import calculate_risk_parity_weights
from src.optimization.robust_mvo import calculate_enhanced_mvo_weights
from src.optimization.performance import calculate_risk_contribution

@pytest.fixture
def sample_opt_inputs() -> dict:
    """Creates sample inputs for optimization functions."""
    tickers = ['ASSET_A', 'ASSET_B', 'ASSET_C']
    mu = pd.Series([0.10, 0.15, 0.12], index=tickers)
    cov_matrix = pd.DataFrame([
        [0.04, 0.015, 0.01],
        [0.015, 0.09, 0.02],
        [0.01, 0.02, 0.0625]
    ], index=tickers, columns=tickers)
    risk_free_rate = 0.02
    return {'mu': mu, 'cov_matrix': cov_matrix, 'risk_free_rate': risk_free_rate}

def test_weights_sum_to_one(sample_opt_inputs):
    """Tests that portfolio weights from all optimizers sum to 1."""
    mu, cov, rf = sample_opt_inputs['mu'], sample_opt_inputs['cov_matrix'], sample_opt_inputs['risk_free_rate']
    
    mvo_weights, _ = calculate_mvo_weights(mu, cov, rf)
    assert np.isclose(mvo_weights.sum(), 1.0)
    
    rp_weights = calculate_risk_parity_weights(cov)
    assert np.isclose(rp_weights.sum(), 1.0)
    
    epsilon = pd.Series([0.01, 0.01, 0.01], index=mu.index)
    enhanced_weights, _ = calculate_enhanced_mvo_weights(
        mu, cov, rf, rp_weights, epsilon, rho_rp=1.0, tau_l2=0.01
    )
    assert np.isclose(enhanced_weights.sum(), 1.0)

def test_long_only_constraint(sample_opt_inputs):
    """Tests that all weights are non-negative."""
    mu, cov, rf = sample_opt_inputs['mu'], sample_opt_inputs['cov_matrix'], sample_opt_inputs['risk_free_rate']
    
    mvo_weights, _ = calculate_mvo_weights(mu, cov, rf)
    assert all(mvo_weights >= 0)
    
    rp_weights = calculate_risk_parity_weights(cov)
    assert all(rp_weights >= 0)

def test_risk_parity_contributions(sample_opt_inputs):
    """Tests that risk parity contributions are nearly equal."""
    cov = sample_opt_inputs['cov_matrix']
    rp_weights = calculate_risk_parity_weights(cov)
    
    contributions = calculate_risk_contribution(rp_weights, cov)
    
    # Check that the standard deviation of contributions is very small
    assert contributions.std() < 1e-4

def test_enhanced_mvo_regularization(sample_opt_inputs):
    """Tests that enhanced MVO weights move towards RP as rho increases."""
    mu, cov, rf = sample_opt_inputs['mu'], sample_opt_inputs['cov_matrix'], sample_opt_inputs['risk_free_rate']
    rp_weights = calculate_risk_parity_weights(cov)
    epsilon = pd.Series([0.01, 0.01, 0.01], index=mu.index)

    # Low regularization
    enhanced_low_rho, _ = calculate_enhanced_mvo_weights(
        mu, cov, rf, rp_weights, epsilon, rho_rp=0.1, tau_l2=0.0
    )
    
    # High regularization
    enhanced_high_rho, _ = calculate_enhanced_mvo_weights(
        mu, cov, rf, rp_weights, epsilon, rho_rp=100.0, tau_l2=0.0
    )
    
    # The L2 norm of the difference between high-rho weights and RP weights
    # should be smaller than for the low-rho weights.
    diff_high = np.linalg.norm(enhanced_high_rho - rp_weights)
    diff_low = np.linalg.norm(enhanced_low_rho - rp_weights)
    
    assert diff_high < diff_low