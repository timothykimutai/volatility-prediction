import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

# --- Local Imports ---
from src.data.fetcher import fetch_price_data
from src.models.garch_model import get_garch_volatility_forecast
from src.optimization.mvo import calculate_mvo_weights, calculate_efficient_frontier
from src.optimization.risk_parity import calculate_risk_parity_weights
from src.optimization.robust_mvo import calculate_enhanced_mvo_weights
from src.optimization.performance import calculate_portfolio_performance
import app.ui_helpers as ui

# --- App Flow ---
def main():
    # Header
    ui.display_header()
    
    # Sidebar parameters
    params = ui.display_sidebar()
    
    # Button to trigger calculations
    if st.button("📊 Calculate Portfolios"):
        with st.spinner("Fetching data and calculating portfolios..."):
            # Fetch prices
            price_data = fetch_price_data(params["tickers"], params["start_date"], params["end_date"])
            if price_data.empty:
                st.error("No price data available for the selected tickers/dates.")
                return
            ui.display_price_chart(price_data)

            # GARCH volatility forecasts
            garch_vol, success_map = get_garch_volatility_forecast(price_data, params["horizon"])
            ui.display_garch_forecasts(garch_vol, success_map)

            # Covariance matrix
            if params["covariance_method"] == "GARCH-based":
                cov_matrix = price_data.pct_change().cov()  # placeholder, ideally use GARCH covariance
            else:
                from sklearn.covariance import LedoitWolf
                returns = price_data.pct_change().dropna()
                cov_matrix = LedoitWolf().fit(returns).covariance_
                cov_matrix = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
            ui.display_covariance_heatmap(cov_matrix, f"Covariance Matrix ({params['covariance_method']})")

            # Expected returns (historical mean for simplicity)
            returns = price_data.pct_change().dropna()
            mu = returns.mean() * 252  # annualized

            # Portfolios
            portfolios = {}

            # Classic MVO
            w_mvo, perf_mvo = calculate_mvo_weights(mu, cov_matrix, params["risk_free_rate"])
            portfolios["Classic MVO"] = {"weights": w_mvo, "perf": perf_mvo}

            # Risk Parity
            w_rp = calculate_risk_parity_weights(cov_matrix)
            portfolios["Risk Parity"] = {
                "weights": w_rp,
                "perf": calculate_portfolio_performance(w_rp, mu, cov_matrix, params["risk_free_rate"])
            }

            # Enhanced MVO
            w_enhanced, perf_enhanced = calculate_enhanced_mvo_weights(
                mu,
                cov_matrix,
                params["risk_free_rate"],
                w_ref=w_rp,
                epsilon=pd.Series(params.get("robust_epsilon", 0), index=mu.index),
                rho_rp=params.get("rp_strength", 0.1),
                tau_l2=params.get("l2_reg", 0.01),
            )
            portfolios["Enhanced MVO"] = {"weights": w_enhanced, "perf": perf_enhanced}

            # Efficient Frontier
            frontier_points = calculate_efficient_frontier(mu, cov_matrix, params["frontier_points"])
            ui.display_efficient_frontier(frontier_points, portfolios)

            # Comparison
            ui.display_portfolio_comparison(portfolios, cov_matrix)

            # Methodology
            ui.display_methodology()
    else:
        st.info("Adjust parameters in the sidebar and click 'Calculate Portfolios' to run the analysis.")


if __name__ == "__main__":
    main()
