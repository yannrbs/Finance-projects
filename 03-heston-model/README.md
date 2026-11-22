# 03 · Heston Stochastic Volatility Model

Implements the Heston (1993) model, where the variance follows a mean-reverting square-root process
correlated with the underlying:

$$
\frac{dS_t}{S_t} = (r - q)\,dt + \sqrt{v_t}\,dW_t^S, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad
d\langle W^S, W^v\rangle_t = \rho\,dt
$$

## What the notebook does

1. **Variance simulation**: Euler scheme with full truncation, so the variance never becomes negative.
2. **Monte Carlo pricing**: correlated Brownian motions built from two independent normals, simulated step by step and vectorised over paths.
3. **Semi-closed-form pricing**: call price from the characteristic function and Fourier inversion, using the numerically stable formulation of Albrecher et al. (*The Little Heston Trap*); put price from put-call parity.
4. **Implied volatility**: Heston prices of out-of-the-money options are inverted through Black-Scholes with Brent's method, giving the smile (ρ = 0), the skew (ρ < 0) and the full surface across strikes and maturities.

## Results

- **Validation**: the Fourier price of an OTM call (K = 115, S₀ = 100, ρ = −0.7) is **7.658**, inside the Monte Carlo
  95% confidence interval [7.559, 7.742]. With ξ → 0 and v₀ = θ, the Heston price (8.3376) converges to Black-Scholes (8.3378).
- **Stochastic vs constant volatility**: the same call is worth 8.338 with constant volatility. Negative spot-vol correlation
  makes large upward moves less likely, so the call is about 9% cheaper under Heston.
- **Skew**: with ρ = −0.6, implied volatility is much higher for low strikes than for high strikes, and the skew flattens with maturity.

## File

- `heston_model.ipynb`
