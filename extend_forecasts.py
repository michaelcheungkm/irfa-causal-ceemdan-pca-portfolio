"""Fill causal CEEMDAN forecasts for the two-year window; keep existing daily cache."""
from __future__ import annotations

from reproduce_revised_study import (
    TEST_END,
    TEST_START,
    TICKERS,
    load_market_data,
    run_walkforward,
)


def main() -> None:
    close, open_, _djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    print(
        f"Two-year window {TEST_START} to {TEST_END}: "
        f"{len(test_index)} sessions, {len(decision_dates)} decisions"
    )
    causal = run_walkforward(close, decision_dates)
    print("Predictors", list(causal), "rows", len(next(iter(causal.values()))))
    idx = next(iter(causal.values())).index
    print(idx.min().date(), "to", idx.max().date())


if __name__ == "__main__":
    main()
