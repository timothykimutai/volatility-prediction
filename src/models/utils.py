from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def calculate_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates logarithmic returns from price data."""
    return np.log(price_data / price_data.shift(1)).dropna()

def calculate_covariance_matrix(
    log_returns: pd.DataFrame,
    use_garch_vol: bool = True,
    garch_vol_forecasts: Optional[Union[pd.Series, np.ndarray]] = None,
    shrinkage: bool = False
) -> pd.DataFrame:
    if use_garch_vol and garch_vol_forecasts is not None:
        # Build covariance from GARCH vol and historical correlation
        corr_matrix = log_returns.corr()
        if shrinkage:
            # Ledoit-Wolf works on covariance, so we apply it to unscaled returns and extract correlation
            lw = LedoitWolf().fit(log_returns)
            # The shrinkage is on the covariance, we need to re-normalize to get correlation
            shrunk_cov = lw.covariance_
            inv_std = np.diag(1 / np.sqrt(np.diag(shrunk_cov)))
            corr_matrix = pd.DataFrame(inv_std @ shrunk_cov @ inv_std, index=log_returns.columns, columns=log_returns.columns)

        # Ensure garch_vol_forecasts is an array aligned to columns
        if isinstance(garch_vol_forecasts, pd.Series):
            vols = garch_vol_forecasts.reindex(log_returns.columns).fillna(0).values
        else:
            vols = np.asarray(garch_vol_forecasts)

        vol_diag = np.diag(vols)
        cov_matrix = pd.DataFrame(
            vol_diag @ corr_matrix.values @ vol_diag,
            index=log_returns.columns,
            columns=log_returns.columns
        )
        return cov_matrix
    else:
        # Use sample covariance (with or without shrinkage)
        if shrinkage:
            lw = LedoitWolf().fit(log_returns * np.sqrt(252))  # Annualize before fitting
            return pd.DataFrame(lw.covariance_, index=log_returns.columns, columns=log_returns.columns)
        else:
            return log_returns.cov() * 252