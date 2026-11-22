# 01 · Monte Carlo Option Pricing

Prices European calls and puts by Monte Carlo simulation under geometric Brownian motion
(with a continuous dividend yield *q*), and checks the result against the Black-Scholes closed form.

## Method

1. Sample terminal prices $S_T$ exactly: $\ln S_T$ is Gaussian under GBM, so no time discretisation is needed.
2. Compute the payoff on each path: `max(S_T − K, 0)` for a call, `max(K − S_T, 0)` for a put.
3. Average the discounted payoffs. The standard error gives a 95% confidence interval, which shrinks at rate $1/\sqrt{n}$.

## Results

S₀ = 130, K = 135, σ = 30%, r = 5%, q = 2%, T = 0.5 year, 100,000 simulations:

| Option | Monte Carlo (95% CI) | Black-Scholes | Gap |
|--------|----------------------|---------------|-----|
| Call | 9.571 [9.464, 9.677] | 9.580 | 0.09% |
| Put | 12.519 [12.426, 12.613] | 12.540 | 0.16% |

## File

- `monte_carlo_pricing.ipynb`
