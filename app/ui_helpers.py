import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.optimization.performance import calculate_risk_contribution

def display_header():
    st.title("🛡️ Enhanced Portfolio Optimizer")
    st.markdown("""
    This application constructs and compares investment portfolios using classical and enhanced optimization techniques.
    It combines **GARCH volatility forecasts** with **robust mean-variance optimization**, regularized by **risk-parity** principles.
    """)

def display_sidebar() -> dict:
    """Creates the sidebar for user inputs and returns a dictionary of parameters."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        tickers_str = st.text_input(
            "Enter Tickers (comma-separated)", 
            "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,JPM,V"
        )
        tickers = [ticker.strip().upper() for ticker in tickers_str.split(",") if ticker.strip()]
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        with col2:
            end_date = st.date_input("End Date", pd.to_datetime("today"))
        
        st.subheader("Model Parameters")
        horizon = st.slider("Forecast Horizon (days)", 1, 252, 63)
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
        frontier_points = st.slider("Efficient Frontier Points", 10, 100, 50)
        
        st.subheader("Enhancements")
        covariance_method = st.selectbox("Covariance Method", ["GARCH-based", "Ledoit-Wolf"])
        
        st.markdown("**Robustness & Regularization**")
        robust_epsilon = st.slider(
            "Return Uncertainty (ε)", 0.0, 2.0, 0.5, 0.05,
            help="Multiplier for the standard error of mean returns to define the uncertainty box."
        )
        rp_strength = st.slider(
            "Risk-Parity Guidance (ρ)", 0.0, 10.0, 1.0, 0.1,
            help="Strength of regularization towards the risk-parity portfolio."
        )
        l2_reg = st.slider(
            "L2 Regularization (τ)", 0.0, 1.0, 1e-5, 1e-6, format="%.6f",
            help="Ridge penalty for numerical stability and diversification."
        )
        
    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "horizon": horizon,
        "risk_free_rate": risk_free_rate,
        "frontier_points": frontier_points,
        "covariance_method": covariance_method,
        "robust_epsilon": robust_epsilon,
        "rp_strength": rp_strength,
        "l2_reg": l2_reg,
    }

def display_price_chart(price_data: pd.DataFrame):
    st.subheader("Normalized Price Performance")
    normalized_prices = (price_data / price_data.iloc[0])
    fig = px.line(normalized_prices, title="Asset Price Evolution (Normalized)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Price", legend_title="Ticker")
    st.plotly_chart(fig, use_container_width=True)

def display_garch_forecasts(garch_vol: pd.Series, success_map: dict):
    st.subheader("GARCH Volatility Forecasts")
    import numpy as np
    df = pd.DataFrame({
        "Ticker": garch_vol.index,
        "Annualized Volatility (%)": np.array(garch_vol.values, dtype=float) * 100,
        "GARCH Fit Status": [ "✅ Success" if success_map[t] else "⚠️ Fallback" for t in garch_vol.index]
    }).set_index("Ticker")
    st.dataframe(df.style.format({"Annualized Volatility (%)": "{:.2f}"}))

def display_covariance_heatmap(cov_matrix: pd.DataFrame, title: str):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=False, cmap="viridis", ax=ax)
    plt.title("Asset Covariance Heatmap")
    st.pyplot(fig)

def display_portfolio_comparison(portfolios: dict, cov_matrix: pd.DataFrame):
    st.subheader("📊 Portfolio Comparison")
    
    # --- Weights Table ---
    weights_df = pd.DataFrame({name: p["weights"] for name, p in portfolios.items()})
    st.markdown("##### Portfolio Weights (%)")
    st.dataframe(weights_df.multiply(100).style.format("{:.2f}").background_gradient(cmap='Greens', axis=1))

    # --- Performance Metrics ---
    perf_df = pd.DataFrame({name: p["perf"] for name, p in portfolios.items()}).T
    perf_df.columns = ["Expected Return", "Volatility", "Sharpe Ratio"]
    st.markdown("##### Performance Metrics")
    st.dataframe(perf_df.style.format({
        "Expected Return": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
    }))

    # --- Risk Contributions ---
    st.markdown("##### Risk Contributions (%)")
    risk_contribs_df = pd.DataFrame()
    for name, p in portfolios.items():
        risk_contribs_df[name] = calculate_risk_contribution(p["weights"], cov_matrix) * 100
    st.bar_chart(risk_contribs_df)
    
    # --- Download Weights ---
    csv = weights_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Weights as CSV",
        data=csv,
        file_name='portfolio_weights.csv',
        mime='text/csv',
    )
    

def display_efficient_frontier(frontier_points: pd.DataFrame, portfolios: dict):
    st.subheader("Efficient Frontier")
    
    fig = go.Figure()
    
    # Plot the frontier
    fig.add_trace(go.Scatter(
        x=frontier_points['Volatility'],
        y=frontier_points['Return'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Plot the portfolios
    colors = px.colors.qualitative.Plotly
    for i, (name, p) in enumerate(portfolios.items()):
        fig.add_trace(go.Scatter(
            x=[p['perf']['volatility']],
            y=[p['perf']['expected_return']],
            mode='markers',
            marker=dict(size=12, color=colors[i], symbol='star'),
            name=name
        ))

    fig.update_layout(
        title="Efficient Frontier with Optimal Portfolios",
        xaxis_title="Annualized Volatility (Standard Deviation)",
        yaxis_title="Annualized Expected Return",
        yaxis_tickformat=".2%",
        xaxis_tickformat=".2%",
        legend_title="Portfolios"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_methodology():
    with st.expander("📘 Methodology & Caveats"):
        st.markdown(r"""
        **1. Volatility Forecasting:**
        - We use a **GARCH(1,1)** model for each asset to capture volatility clustering, which is common in financial markets. This provides a more responsive, forward-looking volatility estimate than historical standard deviation.
        - If a GARCH model fails to converge, we fall back to the asset's sample volatility.

        **2. Covariance Matrix:**
        - The final covariance matrix is constructed by combining GARCH volatility forecasts with the historical correlation matrix.
        - **Ledoit-Wolf Shrinkage** can be applied to stabilize the correlation matrix, especially with many assets.

        **3. Enhanced MVO Objective:**
        The core of the "Enhanced MVO" model is the objective function:
        $$
        \underset{w}{\text{minimize}} \quad \frac{1}{2} w^T \Sigma w - \lambda ( w^T \mu - \sum \epsilon_i |w_i| ) + \frac{\rho}{2} ||w - w_{RP}||_2^2 + \frac{\tau}{2} ||w||_2^2
        $$
        - **$w^T \Sigma w$**: Minimizes portfolio risk.
        - **$( w^T \mu - \sum \epsilon_i |w_i| )$**: Maximizes return while being robust to estimation errors in mean returns ($\mu$). $\epsilon_i$ defines the "uncertainty box" for each asset's expected return.
        - **$||w - w_{RP}||_2^2$**: This term pulls the solution towards the **Risk-Parity** portfolio, promoting better diversification of risk. `ρ` controls its influence.
        - **$||w||_2^2$**: A small L2 penalty (`τ`) ensures numerical stability.

        **Caveats:**
        - **Past performance is not indicative of future results.** All inputs are based on historical data.
        - **Expected returns ($\mu$) are notoriously difficult to forecast.** The robust optimization framework acknowledges this but does not eliminate the problem.
        - This tool is for educational and illustrative purposes and is **not financial advice**.
        """)