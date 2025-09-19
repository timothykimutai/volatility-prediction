import logging
from typing import Tuple, Dict, Optional, Any, Mapping

import numpy as np
import pandas as pd
from arch import arch_model
from joblib import Parallel, delayed

def _fit_single_garch(returns: pd.Series) -> Tuple[str, Optional[Any], float]:
    """Fits a GARCH(1,1) model to a single asset's returns.

    Returns a tuple of (ticker, fit_result_or_None, fallback_annualized_vol).
    The fit result is typed as Any because the arch package returns a custom result object.
    """
    ticker = str(returns.name) if returns.name is not None else "UNKNOWN"
    # GARCH models require non-zero variance
    variance = returns.var()
    if not isinstance(variance, (int, float, np.floating)) or variance < 1e-8:
        logging.warning(f"Skipping GARCH for {ticker}: zero or invalid variance.")
        return ticker, None, float(returns.std() * np.sqrt(252))

    # Scale returns for better convergence
    scaled_returns = returns * 100
    
    try:
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
        result = model.fit(disp='off', show_warning=False)

        # The arch result object exposes ``convergence_flag`` in some versions; fall back safely.
        converged = getattr(result, 'convergence_flag', 0) == 0
        if not converged:
            raise ValueError("GARCH model did not converge.")

        logging.info(f"GARCH model converged for {ticker}.")
        return ticker, result, float('nan')  # Success case, vol will be from forecast
        
    except Exception as e:
        logging.warning(f"GARCH fit failed for {ticker}: {e}. Falling back to sample volatility.")
        # Calculate annualized sample vol as fallback
        fallback_vol = float(returns.std() * np.sqrt(252))
        return ticker, None, fallback_vol

def get_garch_volatility_forecast(
    log_returns: pd.DataFrame, horizon: int = 63
) -> Tuple[pd.Series, Dict[str, bool]]:
    """Compute per-ticker annualized volatility forecasts using GARCH.

    Returns a tuple: (vol_series, success_map) where vol_series maps ticker -> annualized vol
    and success_map maps ticker -> whether GARCH was used (True) or fallback (False).
    """
    tickers = list(log_returns.columns)

    # Run fits in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_fit_single_garch)(log_returns[ticker]) for ticker in tickers
    )

    vol_forecasts: Dict[str, float] = {}
    garch_success_map: Dict[str, bool] = {}

    for ticker, fit_result, fallback_vol in results:
        if fit_result is not None:
            # Forecast and annualize
            forecast = fit_result.forecast(horizon=horizon, reindex=False)
            # Use mean variance forecast over the horizon
            # ``forecast.variance`` can be a DataFrame-like; fall back defensively
            try:
                mean_forecast_var = float(forecast.variance.values.mean())
            except Exception:
                mean_forecast_var = float(np.nan)

            if np.isnan(mean_forecast_var):
                vol_forecasts[ticker] = fallback_vol
                garch_success_map[ticker] = False
            else:
                # Unscale and annualize
                annualized_vol = np.sqrt(mean_forecast_var * 252) / 100
                vol_forecasts[ticker] = float(annualized_vol)
                garch_success_map[ticker] = True
        else:
            # Use the pre-calculated fallback volatility
            vol_forecasts[ticker] = float(fallback_vol)
            garch_success_map[ticker] = False

    return pd.Series(vol_forecasts), garch_success_map