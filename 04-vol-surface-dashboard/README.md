# 04 · Implied Volatility Surface & Greeks Dashboard

Interactive Streamlit dashboard that builds the implied volatility surface of any US-listed stock
from live option chains, and maps the Black-Scholes Greeks across strikes and maturities.

## Pipeline

1. **Market data**: option chains for every listed expiry (yfinance); the 10Y Treasury yield is used as the risk-free rate.
2. **Cleaning**: keep out-of-the-money options (calls above the spot, puts below it), drop quotes with a zero bid or a
   bid-ask spread above a user-defined threshold, and drop maturities under 5 days.
3. **SVI calibration**: on each maturity slice, fit the raw SVI parametrisation of total implied variance

   $$w(k) = a + b\left(\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right), \qquad k = \ln(K/S)$$

   with bounded L-BFGS-B; slices with a poor fit are discarded.
4. **Surface**: interpolate the calibrated slices across maturities on a strike grid concentrated around the money.
5. **Greeks**: compute Delta, Gamma, Theta (per day), Vega (per vol point) and Rho (per 1% rate) with Black-Scholes
   on the calibrated surface, and display each one as an interactive 3D Plotly surface.

## Run it

```bash
pip install -r ../requirements.txt
streamlit run app.py
```

Enter a ticker (e.g. `AAPL`, `TSLA`) in the sidebar, then adjust the spread filter and the option type.

## File

- `app.py`
