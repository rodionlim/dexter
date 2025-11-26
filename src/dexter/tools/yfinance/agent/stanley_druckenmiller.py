from datetime import datetime, timedelta
from typing import Optional

from dexter.tools.yfinance.fundamentals import yf_search_line_items
from dexter.tools.yfinance.prices import yf_get_prices


def analyze_growth_and_momentum(ticker: str) -> dict:
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
    """
    # Fetch financial line items
    financial_line_items = yf_search_line_items(
        ticker=ticker,
        line_items=["revenue", "earnings_per_share"],
        period="annual",
        limit=4,
    )

    # Fetch prices
    end_date = datetime.now().date().isoformat()
    start_date = (datetime.now() - timedelta(days=90)).date().isoformat()
    # yf_get_prices is a tool, so we need to invoke it properly or call the underlying logic if exposed.
    # Since it's a @tool, we can call it directly as a function in Python if we import the decorated function.
    # However, LangChain tools sometimes wrap the function.
    # Let's assume we can call it as a function for now, but handle the potential dict return.
    prices_data = yf_get_prices.invoke(
        {
            "ticker": ticker,
            "interval": "day",
            "interval_multiplier": 1,
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    prices = prices_data.get("prices", [])

    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "Insufficient financial data for growth analysis",
        }

    details = []
    raw_score = 0  # We'll sum up a maximum of 9 raw points, then scale to 0–10

    #
    # 1. Revenue Growth (annualized CAGR)
    #
    revenues = [
        fi.get("revenue")
        for fi in financial_line_items
        if fi.get("revenue") is not None
    ]
    if len(revenues) >= 2:
        latest_rev = revenues[0]
        older_rev = revenues[-1]
        num_years = len(revenues) - 1

        # Ensure values are numbers
        if isinstance(latest_rev, (int, float)) and isinstance(older_rev, (int, float)):
            if older_rev > 0 and latest_rev > 0:
                # CAGR formula: (ending_value/beginning_value)^(1/years) - 1
                rev_growth = (latest_rev / older_rev) ** (1 / num_years) - 1
                if rev_growth > 0.08:  # 8% annualized (adjusted for CAGR)
                    raw_score += 3
                    details.append(
                        f"Strong annualized revenue growth: {rev_growth:.1%}"
                    )
                elif rev_growth > 0.04:  # 4% annualized
                    raw_score += 2
                    details.append(
                        f"Moderate annualized revenue growth: {rev_growth:.1%}"
                    )
                elif rev_growth > 0.01:  # 1% annualized
                    raw_score += 1
                    details.append(
                        f"Slight annualized revenue growth: {rev_growth:.1%}"
                    )
                else:
                    details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
            else:
                details.append(
                    "Older revenue is zero/negative; can't compute revenue growth."
                )
        else:
            details.append("Invalid revenue data types.")
    else:
        details.append("Not enough revenue data points for growth calculation.")

    #
    # 2. EPS Growth (annualized CAGR)
    #
    eps_values = [
        fi.get("earnings_per_share")
        for fi in financial_line_items
        if fi.get("earnings_per_share") is not None
    ]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        num_years = len(eps_values) - 1

        if isinstance(latest_eps, (int, float)) and isinstance(older_eps, (int, float)):
            # Calculate CAGR for positive EPS values
            if older_eps > 0 and latest_eps > 0:
                # CAGR formula for EPS
                eps_growth = (latest_eps / older_eps) ** (1 / num_years) - 1
                if eps_growth > 0.08:  # 8% annualized (adjusted for CAGR)
                    raw_score += 3
                    details.append(f"Strong annualized EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.04:  # 4% annualized
                    raw_score += 2
                    details.append(f"Moderate annualized EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.01:  # 1% annualized
                    raw_score += 1
                    details.append(f"Slight annualized EPS growth: {eps_growth:.1%}")
                else:
                    details.append(
                        f"Minimal/negative annualized EPS growth: {eps_growth:.1%}"
                    )
            else:
                details.append(
                    "Older EPS is near zero; skipping EPS growth calculation."
                )
        else:
            details.append("Invalid EPS data types.")
    else:
        details.append("Not enough EPS data points for growth calculation.")

    #
    # 3. Price Momentum
    #
    # We'll give up to 3 points for strong momentum
    if prices and len(prices) > 30:
        # Prices are already sorted by date in yf_get_prices
        close_prices = [p.get("close") for p in prices if p.get("close") is not None]
        if len(close_prices) >= 2:
            start_price = close_prices[0]
            end_price = close_prices[-1]
            if start_price > 0:
                pct_change = (end_price - start_price) / start_price
                if pct_change > 0.50:
                    raw_score += 3
                    details.append(f"Very strong price momentum: {pct_change:.1%}")
                elif pct_change > 0.20:
                    raw_score += 2
                    details.append(f"Moderate price momentum: {pct_change:.1%}")
                elif pct_change > 0:
                    raw_score += 1
                    details.append(f"Slight positive momentum: {pct_change:.1%}")
                else:
                    details.append(f"Negative price momentum: {pct_change:.1%}")
            else:
                details.append("Invalid start price (<= 0); can't compute momentum.")
        else:
            details.append("Insufficient price data for momentum calculation.")
    else:
        details.append("Not enough recent price data for momentum analysis.")

    # We assigned up to 3 points each for:
    #   revenue growth, eps growth, momentum
    # => max raw_score = 9
    # Scale to 0–10
    final_score = min(10, (raw_score / 9) * 10)

    return {"score": final_score, "details": "; ".join(details)}
