import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Dashboard - Volatility & Greeks", layout="wide")
st.title("Implied Volatility Surface & Greeks Dashboard for US Options")

# =============================================================================
# MARKET DATA AND RISK-FREE RATE
# =============================================================================

@st.cache_data(show_spinner=False)
def get_us_risk_free_rate() -> float:
    """10-year US Treasury yield (in %), with a 4.5% fallback if the data is unavailable."""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 4.5


@st.cache_data(show_spinner=False)
def fetch_us_options_data(ticker_symbol: str, spot: float) -> pd.DataFrame:
    """Out-of-the-money calls and puts with a positive implied volatility, for every listed expiry."""
    ticker = yf.Ticker(ticker_symbol)
    try:
        expirations = ticker.options
    except Exception:
        return pd.DataFrame()

    if not expirations:
        return pd.DataFrame()

    data_frames = []
    today = datetime.today()

    for exp_date in expirations:
        try:
            opt_chain = ticker.option_chain(exp_date)

            # Out-of-the-money options are the most liquid: calls above the spot, puts below it
            calls_otm = opt_chain.calls[opt_chain.calls["strike"] >= spot]
            puts_otm = opt_chain.puts[opt_chain.puts["strike"] < spot]
            df_merged = pd.concat([calls_otm, puts_otm], ignore_index=True)

            exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
            T = np.maximum((exp_datetime - today).days / 365.0, 1e-4)

            df_merged["T"] = T
            df_merged = df_merged[df_merged["impliedVolatility"] > 0.01]

            if not df_merged.empty:
                data_frames.append(df_merged[["strike", "T", "impliedVolatility", "bid", "ask"]])
        except Exception:
            continue

    if not data_frames:
        return pd.DataFrame()

    # Concatenate all maturities; if a (strike, T) pair appears twice, keep the median values
    df_final = pd.concat(data_frames, ignore_index=True)
    return df_final.groupby(["strike", "T"]).median().reset_index()


# =============================================================================
# SVI CALIBRATION
# =============================================================================
def svi_total_variance(params, k):
    """Raw SVI parametrisation of total implied variance w(k) = sigma_impl^2 * T, with k = ln(K/S)."""
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def calibrate_svi_slice(k_market, w_market):
    """Least-squares fit of the SVI parameters on one maturity slice. Returns (params, error)."""
    def objective(params):
        w_model = svi_total_variance(params, k_market)
        if np.any(w_model < 1e-5):  # total variance must stay positive
            return 1e10
        return np.sum((w_model - w_market) ** 2)

    p0 = [np.min(w_market) / 2, 0.1, 0.0, 0.0, 0.1]
    bounds = ((1e-5, 2.0), (1e-5, 0.5), (-0.99, 0.99), (-1.0, 1.0), (1e-3, 2.0))
    res = minimize(objective, p0, bounds=bounds, method="L-BFGS-B")
    return res.x, res.fun


# =============================================================================
# GREEKS (BLACK-SCHOLES)
# =============================================================================
class BlackScholesModel:
    def __init__(self, S: float, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray, option_type: str = "Call"):
        self.S = S
        self.K = K
        self.T = np.maximum(T, 1e-5)
        self.r = r
        self.sigma = np.maximum(sigma, 1e-4)
        self.option_type = option_type.lower()

        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
                  (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def delta(self) -> np.ndarray:
        if self.option_type == "call":
            return norm.cdf(self.d1)
        return norm.cdf(self.d1) - 1.0

    def gamma(self) -> np.ndarray:
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self) -> np.ndarray:
        """Theta per calendar day."""
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == "call":
            raw = term1 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            raw = term1 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return raw / 365.0

    def vega(self) -> np.ndarray:
        """Vega per 1 volatility point (1%)."""
        return (self.S * norm.pdf(self.d1) * np.sqrt(self.T)) / 100.0

    def rho(self) -> np.ndarray:
        """Rho per 1% change in the risk-free rate."""
        if self.option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 100.0
        return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 100.0


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("Settings")

ticker_input = st.sidebar.text_input("US ticker", value="AAPL").strip().upper()
us_rate_detected = get_us_risk_free_rate()
r = us_rate_detected / 100.0
st.sidebar.caption(f"Risk-free rate (US 10Y Treasury): {us_rate_detected:.2f}%")

st.sidebar.markdown("---")
st.sidebar.subheader("Liquidity filters")

k_spread = st.sidebar.number_input(
    "Max bid-ask spread ($)",
    min_value=0.001,
    max_value=50.0,
    value=0.50,
    step=0.10,
    help="Discards illiquid options whose bid-ask spread is above this value."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Greeks settings")
option_type = st.sidebar.radio("Option type for the Greeks", ["Call", "Put"])

# =============================================================================
# DATA VALIDATION AND SVI VOLATILITY SURFACE
# =============================================================================
if not ticker_input:
    st.warning("Please enter a valid ticker.")
    st.stop()

with st.spinner(f"Calibrating the SVI surface for {ticker_input}..."):
    try:
        stock = yf.Ticker(ticker_input)
        hist = stock.history(period="1d")
        if hist.empty:
            st.error(f"Ticker '{ticker_input}' not found.")
            st.stop()

        close_val = hist["Close"].iloc[-1]
        if pd.isna(close_val) or close_val <= 0:
            st.error(f"The spot price of '{ticker_input}' is unavailable or invalid (market closed?).")
            st.stop()
        S_val = float(close_val)
        st.sidebar.success(f"{ticker_input} spot price: ${S_val:.2f}")

        df_options = fetch_us_options_data(ticker_input, S_val)

        if df_options.empty:
            st.error(f"No option data available for '{ticker_input}'.")
            st.stop()

        df_options["spread"] = df_options["ask"] - df_options["bid"]
        df_options = df_options[(df_options["bid"] > 0) & (df_options["spread"] <= k_spread)]

        min_T_stable = 5.0 / 365.0   # shortest maturity kept for the Greeks
        min_T_vol = 7.0 / 365.0      # shortest maturity shown on the volatility surface

        df_options = df_options[df_options["T"] >= min_T_stable]

        if len(df_options) < 5:
            st.error(f"Not enough data left after applying the ${k_spread} spread filter.")
            st.stop()

        # Strike grid built from the strikes actually traded on the market
        min_market_strike = df_options["strike"].min()
        max_market_strike = df_options["strike"].max()

        # Lower bound: the market minimum, but never below 10% of the spot (to keep ln(K/S) well defined)
        # and never above 70% of the spot (so that the grid stays increasing)
        k_min = min(max(min_market_strike, S_val * 0.10), S_val * 0.70)
        # Upper bound: the market maximum, but at least 130% of the spot
        k_max = max(max_market_strike, S_val * 1.30)

        # Denser grid around the money, where the options are the most liquid
        k_low = np.linspace(k_min, S_val * 0.85, 25, endpoint=False)
        k_atm = np.linspace(S_val * 0.85, S_val * 1.15, 50, endpoint=False)
        k_high = np.linspace(S_val * 1.15, k_max, 25)

        grid_k = np.concatenate([k_low, k_atm, k_high])
        k_dense = np.log(grid_k / S_val)

        market_T_list = []
        iv_matrix = []
        unique_T = np.sort(df_options["T"].unique())

        for T_val in unique_T:
            slice_data = df_options[df_options["T"] == T_val]

            # The SVI calibration uses the liquid zone only: strikes between 60% and 140% of the spot
            calib_data = slice_data[
                (slice_data["strike"] >= S_val * 0.6) &
                (slice_data["strike"] <= S_val * 1.4)
            ]
            if len(calib_data) < 5:
                continue

            k_market = np.log(calib_data["strike"] / S_val).values
            w_market = (calib_data["impliedVolatility"] ** 2 * T_val).values

            params, fit_error = calibrate_svi_slice(k_market, w_market)
            if fit_error > 1e-4:  # poor fit: the slice is discarded
                continue

            w_svi_dense = np.maximum(svi_total_variance(params, k_dense), 1e-8)
            iv_svi_dense = np.sqrt(w_svi_dense / T_val)

            # Volatility capped at 500% so that the far wings are displayed without being flattened
            iv_svi_dense = np.clip(iv_svi_dense, 0.01, 5.0)

            market_T_list.append(T_val)
            iv_matrix.append(iv_svi_dense)

        if len(market_T_list) < 2:
            st.error(
                "The spread filter is too strict: "
                "not enough liquid maturities to build the surface."
            )
            st.stop()

        market_T_list = np.array(market_T_list)
        iv_matrix = np.array(iv_matrix)

        max_T_market = market_T_list[-1]

        if max_T_market <= min_T_vol:
            st.error(
                f"The longest available maturity ({max_T_market:.2f} years) is below "
                "the display threshold of the surface (1 week). "
                "Increase the spread tolerance."
            )
            st.stop()

        # Linear interpolation of the calibrated slices across maturities
        interp_func = interp1d(
            market_T_list,
            iv_matrix,
            axis=0,
            bounds_error=False,
            fill_value=(iv_matrix[0], iv_matrix[-1])
        )

        grid_t_greeks = np.linspace(min_T_stable, max_T_market, 40)
        grid_t_vol = np.linspace(min_T_vol, max_T_market, 40)

        vol_surface_greeks = interp_func(grid_t_greeks)
        vol_surface_vol = interp_func(grid_t_vol)

        K_grid_greeks, T_grid_greeks = np.meshgrid(grid_k, grid_t_greeks)
        K_grid_vol, T_grid_vol = np.meshgrid(grid_k, grid_t_vol)

    except Exception as e:
        st.error(f"A technical error occurred: {e}")
        st.stop()

# =============================================================================
# GREEKS WITH THE BLACK-SCHOLES MODEL
# =============================================================================
bsm = BlackScholesModel(
    S=S_val,
    K=K_grid_greeks,
    T=T_grid_greeks,
    r=r,
    sigma=vol_surface_greeks,
    option_type=option_type
)

greeks_dict = {
    "Delta (Δ)": bsm.delta,
    "Gamma (Γ)": bsm.gamma,
    "Theta (Θ) [$/day]": bsm.theta,
    "Vega (ν) [$ per 1% vol]": bsm.vega,
    "Rho (ρ) [$ per 1% rate]": bsm.rho,
}

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["Greeks analysis", "SVI implied volatility surface"])

# --- TAB 1: GREEKS ---
with tab1:
    st.header(f"Sensitivity analysis: Greeks ({option_type})")
    selected_metric = st.selectbox(
        "Choose the Greek to map in 3D:",
        list(greeks_dict.keys())
    )

    metric_surface = greeks_dict[selected_metric]()

    fig_metric = go.Figure(data=[go.Surface(
        z=metric_surface,
        x=K_grid_greeks,
        y=T_grid_greeks,
        colorscale="Plasma"
    )])
    fig_metric.update_layout(
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (T, years)",
            zaxis_title=selected_metric,
            aspectratio=dict(x=1.2, y=1.2, z=0.8)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=850
    )
    st.plotly_chart(fig_metric, use_container_width=True)

# --- TAB 2: VOLATILITY SURFACE ---
with tab2:
    st.header("SVI implied volatility surface (maturities > 1 week)")

    if k_spread <= 1.0:
        st.info(
            "💡 **Strict threshold**: only very liquid short-dated options are shown. "
            "Increase the spread threshold to include longer maturities (LEAPS)."
        )
    else:
        st.info(
            f"💡 **Wide threshold (${k_spread})**: long maturities are included in the surface, "
            "accepting the wider spreads that come with longer-dated options."
        )

    fig_vol = go.Figure(data=[go.Surface(
        z=vol_surface_vol,
        x=K_grid_vol,
        y=T_grid_vol,
        colorscale="Viridis"
    )])
    fig_vol.update_layout(
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (T, years)",
            zaxis_title="Implied volatility (σ)",
            aspectratio=dict(x=1.2, y=1.2, z=0.8)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=850
    )
    st.plotly_chart(fig_vol, use_container_width=True)
