import pandas as pd
import numpy as np
import pytest
from src.models.garch_model import get_garch_volatility_forecast

@pytest.fixture
def sample_returns_data() -> pd.DataFrame:
    """Creates a sample returns DataFrame for testing."""
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=500, freq='B'))
    # Asset with clear GARCH effects
    garch_returns = np.random.normal(0, 0.01, 500)
    for i in range(1, 500):
        sigma2 = 0.0001 + 0.1 * garch_returns[i-1]**2 + 0.85 * (0.01**2)
        garch_returns[i] = np.random.normal(0, np.sqrt(sigma2))

    # Asset with near-zero variance (should fallback)
    flat_returns = np.zeros(500)
    
    # Normal asset (may or may not converge, but should produce a value)
    normal_returns = np.random.normal(0.0005, 0.02, 500)
    
    return pd.DataFrame({
        'GARCH_ASSET': garch_returns,
        'FLAT_ASSET': flat_returns,
        'NORMAL_ASSET': normal_returns
    }, index=dates)

def test_garch_forecast_output_shape_and_type(sample_returns_data):
    """Tests that the GARCH forecast function returns correct types and shapes."""
    vol_forecasts, success_map = get_garch_volatility_forecast(sample_returns_data, horizon=30)
    
    assert isinstance(vol_forecasts, pd.Series)
    assert isinstance(success_map, dict)
    assert len(vol_forecasts) == sample_returns_data.shape[1]
    assert set(vol_forecasts.index) == set(sample_returns_data.columns)
    assert all(isinstance(v, bool) for v in success_map.values())

def test_garch_fallback_mechanism(sample_returns_data):
    """Tests that GARCH correctly falls back to sample vol for problematic data."""
    vol_forecasts, success_map = get_garch_volatility_forecast(sample_returns_data, horizon=30)
    
    # The flat asset should fail GARCH and fall back
    assert not success_map['FLAT_ASSET']
    
    # The fallback volatility should be very close to the sample volatility (annualized)
    expected_fallback_vol = sample_returns_data['FLAT_ASSET'].std() * np.sqrt(252)
    assert np.isclose(vol_forecasts['FLAT_ASSET'], expected_fallback_vol)

def test_garch_success_case(sample_returns_data):
    """Tests a successful GARCH fit."""
    vol_forecasts, success_map = get_garch_volatility_forecast(sample_returns_data, horizon=30)
    
    # The GARCH asset is designed to converge
    assert success_map['GARCH_ASSET']
    # The forecasted volatility should be a reasonable positive number
    assert vol_forecasts['GARCH_ASSET'] > 0
    assert not np.isnan(vol_forecasts['GARCH_ASSET'])