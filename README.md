# Finance Projects

Personal projects in option pricing, Greeks and volatility modeling, built in Python alongside my
Applied Mathematics engineering degree at **INSA Rennes** (France).

| # | Project | What it covers | Tools |
|---|---------|----------------|-------|
| 01 | [Monte Carlo Option Pricing](01-monte-carlo-pricing) | European call/put pricing by simulation, with confidence intervals, benchmarked against the Black-Scholes closed form | NumPy, SciPy |
| 02 | [Black-Scholes Greeks](02-black-scholes-greeks) | Delta, Gamma, Theta, Vega and Rho along a simulated price path | NumPy, Matplotlib |
| 03 | [Heston Stochastic Volatility Model](03-heston-model) | Monte Carlo simulation, semi-closed-form Fourier pricing, implied volatility smile, skew and surface | NumPy, SciPy |
| 04 | [Implied Volatility Surface & Greeks Dashboard](04-vol-surface-dashboard) | Live US option chains, SVI calibration per maturity, 3D volatility and Greeks surfaces | Streamlit, Plotly, yfinance |

## Getting started

```bash
git clone https://github.com/yannrbs/Finance-projects.git
cd Finance-projects
pip install -r requirements.txt

# Notebooks (projects 01 to 03)
jupyter notebook

# Dashboard (project 04)
streamlit run 04-vol-surface-dashboard/app.py
```

## About me

Yann Robcis, Applied Mathematics engineering student at INSA Rennes, looking for a 2 to 3-month
Global Markets internship from June 2027 (Trading, Structuring, Brokerage or Quantitative Analysis).

[LinkedIn](https://www.linkedin.com/in/yann-robcis-126890378) · yann.robcis@gmail.com
