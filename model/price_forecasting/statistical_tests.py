"""Statistical robustness tests for trading strategy evaluation.

Provides:
- Bootstrap confidence intervals for Sharpe ratio
- Permutation test for signal significance
- Rolling Sharpe ratio for regime analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def bootstrap_sharpe_ci(
    pnl_series: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    days_per_year: int = 252,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for the annualized Sharpe ratio.

    Args:
        pnl_series: Array of daily PnL values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        days_per_year: Trading days per year for annualization.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary with point estimate, lower/upper CI bounds, and std error.
    """
    rng = np.random.RandomState(random_seed)
    pnl = np.asarray(pnl_series, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    n = len(pnl)

    if n < 10:
        return {
            "sharpe": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
            "n_observations": n,
        }

    # Point estimate
    sharpe_point = (pnl.mean() / pnl.std()) * np.sqrt(days_per_year) if pnl.std() > 0 else 0.0

    # Bootstrap
    bootstrap_sharpes = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(pnl, size=n, replace=True)
        if sample.std() > 0:
            bootstrap_sharpes[i] = (sample.mean() / sample.std()) * np.sqrt(days_per_year)
        else:
            bootstrap_sharpes[i] = 0.0

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))

    return {
        "sharpe": sharpe_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": bootstrap_sharpes.std(),
        "n_observations": n,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }


def permutation_test_signal(
    signals: np.ndarray,
    returns: np.ndarray,
    n_permutations: int = 5000,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Test whether the trading signal has predictive power above random.

    Permutes the signal-return alignment and computes the fraction of
    permuted Sharpe ratios that exceed the observed Sharpe.

    Args:
        signals: Array of daily trading signals.
        returns: Array of daily returns (or PnL per unit).
        n_permutations: Number of random permutations.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary with observed Sharpe, p-value, and distribution stats.
    """
    rng = np.random.RandomState(random_seed)
    signals = np.asarray(signals, dtype=float)
    returns = np.asarray(returns, dtype=float)

    # Remove NaNs
    valid = ~(np.isnan(signals) | np.isnan(returns))
    signals = signals[valid]
    returns = returns[valid]
    n = len(signals)

    if n < 10:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "n_observations": n}

    # Observed: PnL = signal * return
    observed_pnl = signals * returns
    observed_sharpe = (
        (observed_pnl.mean() / observed_pnl.std()) * np.sqrt(252)
        if observed_pnl.std() > 0
        else 0.0
    )

    # Permutation distribution
    permuted_sharpes = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_signals = rng.permutation(signals)
        perm_pnl = perm_signals * returns
        if perm_pnl.std() > 0:
            permuted_sharpes[i] = (perm_pnl.mean() / perm_pnl.std()) * np.sqrt(252)

    # Two-sided p-value
    p_value = (np.abs(permuted_sharpes) >= np.abs(observed_sharpe)).mean()

    return {
        "observed_sharpe": observed_sharpe,
        "p_value": p_value,
        "permuted_mean": permuted_sharpes.mean(),
        "permuted_std": permuted_sharpes.std(),
        "permuted_95th": np.percentile(np.abs(permuted_sharpes), 95),
        "n_observations": n,
        "n_permutations": n_permutations,
        "significant_5pct": p_value < 0.05,
    }


def rolling_sharpe(
    pnl_series: np.ndarray,
    window: int = 63,
    days_per_year: int = 252,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Args:
        pnl_series: Array or Series of daily PnL values.
        window: Rolling window in trading days (default 63 ~ 3 months).
        days_per_year: Trading days per year for annualization.

    Returns:
        pd.Series with rolling Sharpe values.
    """
    pnl = pd.Series(pnl_series, dtype=float)
    rolling_mean = pnl.rolling(window, min_periods=window // 2).mean()
    rolling_std = pnl.rolling(window, min_periods=window // 2).std()
    rolling_sr = (rolling_mean / rolling_std) * np.sqrt(days_per_year)
    rolling_sr = rolling_sr.replace([np.inf, -np.inf], np.nan)
    return rolling_sr


def print_robustness_report(
    pnl_series: np.ndarray,
    signals: np.ndarray = None,
    returns: np.ndarray = None,
) -> Dict:
    """Print a complete robustness report for a trading strategy.

    Args:
        pnl_series: Daily PnL values.
        signals: Trading signals (optional, for permutation test).
        returns: Underlying returns (optional, for permutation test).

    Returns:
        Dictionary with all test results.
    """
    results = {}

    # Bootstrap CI
    bootstrap = bootstrap_sharpe_ci(pnl_series)
    results["bootstrap"] = bootstrap

    print("=" * 70)
    print("STATISTICAL ROBUSTNESS REPORT")
    print("=" * 70)
    print(f"\n  Bootstrap Sharpe Ratio ({bootstrap['ci_level']*100:.0f}% CI):")
    print(f"    Point estimate: {bootstrap['sharpe']:.3f}")
    print(f"    CI: [{bootstrap['ci_lower']:.3f}, {bootstrap['ci_upper']:.3f}]")
    print(f"    Std error: {bootstrap['std_error']:.3f}")
    print(f"    N observations: {bootstrap['n_observations']}")

    # Permutation test
    if signals is not None and returns is not None:
        perm = permutation_test_signal(signals, returns)
        results["permutation"] = perm

        print(f"\n  Permutation Test (signal significance):")
        print(f"    Observed Sharpe: {perm['observed_sharpe']:.3f}")
        print(f"    p-value: {perm['p_value']:.4f}")
        print(f"    Significant at 5%: {'YES' if perm['significant_5pct'] else 'NO'}")
        print(f"    Random 95th percentile: {perm['permuted_95th']:.3f}")

    # Rolling Sharpe summary
    rolling_sr = rolling_sharpe(pnl_series)
    results["rolling_sharpe_mean"] = rolling_sr.mean()
    results["rolling_sharpe_std"] = rolling_sr.std()
    results["pct_time_positive_sharpe"] = (rolling_sr > 0).mean() * 100

    print(f"\n  Rolling Sharpe (63-day window):")
    print(f"    Mean: {rolling_sr.mean():.3f}")
    print(f"    Std:  {rolling_sr.std():.3f}")
    print(f"    % time positive: {(rolling_sr > 0).mean()*100:.1f}%")
    print("=" * 70)

    return results
