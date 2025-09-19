# 🛡️ Enhanced Portfolio Optimizer

This project provides a production-ready Streamlit application for advanced portfolio construction. It enhances classical Mean-Variance Optimization (MVO) by incorporating GARCH-based volatility forecasts, robust optimization techniques, and risk-parity regularization to create more stable and diversified portfolios.

## ✨ Features

- **Dynamic Volatility Forecasting**: Utilizes univariate **GARCH(1,1)** models for each asset to capture time-varying volatility.
- **Robust Covariance**: Employs **Ledoit-Wolf shrinkage** on the correlation matrix to ensure it is well-conditioned.
- **Three Optimization Models**:
    1.  **Classical MVO**: Maximizes the Sharpe Ratio based on historical means and the forecast covariance.
    2.  **Risk-Parity (RP)**: Constructs a reference portfolio where each asset contributes equally to total risk.
    3.  **Enhanced Robust MVO**: A sophisticated model that minimizes risk while being robust to errors in expected returns and regularized towards the risk-parity solution.
- **Interactive UI**: A user-friendly Streamlit dashboard to configure parameters, run optimizations, and visualize results.
- **Production-Ready**: Built with a modular structure, unit tests, type checking, CI/CD pipeline, and Docker containerization.

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Pandas, NumPy, SciPy
- **Frontend**: Streamlit
- **Financial Modeling**: `arch` (for GARCH), `scikit-learn` (for Ledoit-Wolf), `yfinance` (for data)
- **Visualization**: Plotly, Matplotlib/Seaborn
- **Tooling**: Docker, GitHub Actions, Pytest, MyPy, Ruff, Black

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### 1. Clone the Repository

```bash
git clone [https://github.com/timothykimutai/volatility-prediction.git](https://github.com/timothykimutai/volatility-prediction.git)
cd volatility-prediction
```

### 2. Local Installation (Virtual Environment)

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies using the Makefile
make install
```

### 3. Running the Application

#### Locally

To run the Streamlit application directly:

```bash
make run-app
```

Navigate to `http://localhost:8501` in your web browser.

#### Using Docker

This is the recommended method for a clean, reproducible environment.

```bash
# 1. Build the Docker image
make build-docker

# 2. Run the container
make run-docker
```
The application will be available at `http://localhost:8501`.

## ✅ Code Quality & Testing

This project emphasizes code quality and robustness.

- **Linting**: `make lint` (using Ruff)
- **Formatting**: `make format` (using Black)
- **Type Checking**: `mypy . --ignore-missing-imports`
- **Unit Testing**: `make test` (using Pytest)

To run all checks simultaneously:

```bash
make check
```

## 📂 Project Structure

```
/project-root
├── app/             # Streamlit UI code
├── src/             # Core backend modules (data, models, optimization)
│   ├── data/
│   ├── models/
│   ├── optimization/
│   └── tests/
├── ci/              # GitHub Actions CI/CD workflow
├── docker/          # Docker configuration
├── notebooks/       # Demo Jupyter notebook
├── Makefile         # Shortcuts for common commands
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## 📜 Methodology

The core of this application is the **Enhanced MVO** model, which optimizes the following objective function:

$$
\underset{w}{\text{minimize}} \quad \frac{1}{2} w^T \Sigma w - \lambda \left( w^T \mu - \sum \epsilon_i |w_i| \right) + \frac{\rho}{2} ||w - w_{RP}||_2^2 + \frac{\tau}{2} ||w||_2^2
$$

- **$w^T \Sigma w$**: The risk term (portfolio variance).
- **$( w^T \mu - \sum \epsilon_i |w_i| )$**: A robust return term that accounts for uncertainty (`ε`) in expected returns (`μ`).
- **$||w - w_{RP}||_2^2$**: A regularization term that guides the solution towards a diversified Risk-Parity portfolio (`w_RP`).
- **$||w||_2^2$**: An L2 (ridge) penalty for numerical stability.

For a detailed explanation, please see the **Design Document**.

## 🔮 Future Work

- [ ] Implement multivariate DCC-GARCH for dynamic correlation forecasting.
- [ ] Add a walk-forward backtesting module to compare strategy performance over time.
- [ ] Incorporate transaction cost models and turnover constraints.
- [ ] Deploy the application as a persistent web service.