import statistics

from datetime import datetime, timedelta

from dexter.tools.yfinance.fundamentals import yf_search_line_items
from dexter.tools.yfinance.prices import yf_get_prices, yf_get_price_snapshot


def stanley_druckenmiller_agent(tickers: list[str]) -> dict:
    analysis_data = {}
    for ticker in tickers:
        # Fetch financial line items
        financial_line_items = yf_search_line_items(
            ticker=ticker,
            line_items=[
                "revenue",
                "earnings_per_share",
                "total_debt",
                "shareholders_equity",
                "net_income",
                "free_cash_flow",
                "ebit",
                "ebitda",
                "cash_and_equivalents",
            ],
            period="annual",
            limit=4,
        )

        # Fetch prices
        end_date = datetime.now().date().isoformat()
        start_date = (datetime.now() - timedelta(days=365)).date().isoformat()
        prices_data = yf_get_prices.invoke(
            {
                "ticker": ticker,
                "interval": "day",
                "interval_multiplier": 1,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

        # Fetch market cap
        snapshot_data = yf_get_price_snapshot.invoke({"tickers": [ticker]})
        market_cap = snapshot_data.get(ticker, {}).get("snapshot", {}).get("market_cap")

        prices = prices_data.get("prices", [])

        growth_momentum_analysis = analyze_growth_and_momentum(
            financial_line_items, prices
        )
        risk_reward_analysis = analyze_risk_reward(financial_line_items, prices)
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        total_score = (
            growth_momentum_analysis.get("score", 0)
            + risk_reward_analysis.get("score", 0)
            + valuation_analysis.get("score", 0)
        )
        max_possible_score = 30

        if total_score >= 22:
            signal = "Strong Buy"
        elif total_score >= 16:
            signal = "Buy"
        elif total_score >= 10:
            signal = "Hold"
        else:
            signal = "Sell"

        analysis_data[ticker] = {
            "signal": signal,
            "score": float(f"{total_score:.1f}"),
            "max_score": max_possible_score,
            "growth_momentum_analysis": growth_momentum_analysis,
            "sentiment_analysis": None,
            "insider_activity": None,
            "risk_reward_analysis": risk_reward_analysis,
            "valuation_analysis": valuation_analysis,
        }

    return analysis_data


def analyze_growth_and_momentum(financial_line_items: list, prices: list) -> dict:
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
    """
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

    return {"score": float(f"{final_score:.2f}"), "details": "; ".join(details)}


def analyze_risk_reward(financial_line_items: list, prices: list) -> dict:
    """
    Assesses risk via:
      - Debt-to-Equity
      - Price Volatility
    Aims for strong upside with contained downside.
    """
    if not financial_line_items or not prices:
        return {"score": 0, "details": "Insufficient data for risk-reward analysis"}

    details = []
    raw_score = 0  # We'll accumulate up to 6 raw points, then scale to 0-10

    #
    # 1. Debt-to-Equity
    #
    debt_values = [
        fi.get("total_debt")
        for fi in financial_line_items
        if fi.get("total_debt") is not None
    ]
    equity_values = [
        fi.get("shareholders_equity")
        for fi in financial_line_items
        if fi.get("shareholders_equity") is not None
    ]

    if (
        debt_values
        and equity_values
        and len(debt_values) == len(equity_values)
        and len(debt_values) > 0
    ):
        recent_debt = debt_values[0]
        recent_equity = equity_values[0] if equity_values[0] else 1e-9
        de_ratio = recent_debt / recent_equity
        if de_ratio < 0.3:
            raw_score += 3
            details.append(f"Low debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 0.7:
            raw_score += 2
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 1.5:
            raw_score += 1
            details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f}")
    else:
        details.append("No consistent debt/equity data available.")

    #
    # 2. Price Volatility
    #
    if len(prices) > 10:
        # Prices are typically sorted by date in yf_get_prices, but let's ensure or assume
        # The user code sorted them.
        # prices is a list of dicts.
        # We need to parse 'time' if we want to sort, or assume they are sorted.
        # yf_get_prices returns sorted by date ascending usually.
        # But let's just extract closes.
        close_prices = [p.get("close") for p in prices if p.get("close") is not None]

        if len(close_prices) > 10:
            daily_returns = []
            for i in range(1, len(close_prices)):
                prev_close = close_prices[i - 1]
                if prev_close > 0:
                    daily_returns.append((close_prices[i] - prev_close) / prev_close)
            if daily_returns:
                stdev = statistics.pstdev(daily_returns)  # population stdev
                if stdev < 0.01:
                    raw_score += 3
                    details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.02:
                    raw_score += 2
                    details.append(
                        f"Moderate volatility: daily returns stdev {stdev:.2%}"
                    )
                elif stdev < 0.04:
                    raw_score += 1
                    details.append(f"High volatility: daily returns stdev {stdev:.2%}")
                else:
                    details.append(
                        f"Very high volatility: daily returns stdev {stdev:.2%}"
                    )
            else:
                details.append("Insufficient daily returns data for volatility calc.")
        else:
            details.append(
                "Not enough close-price data points for volatility analysis."
            )
    else:
        details.append("Not enough price data for volatility analysis.")

    # raw_score out of 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": float(f"{final_score:.2f}"), "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [
        fi.get("net_income")
        for fi in financial_line_items
        if fi.get("net_income") is not None
    ]
    fcf_values = [
        fi.get("free_cash_flow")
        for fi in financial_line_items
        if fi.get("free_cash_flow") is not None
    ]
    ebit_values = [
        fi.get("ebit") for fi in financial_line_items if fi.get("ebit") is not None
    ]
    ebitda_values = [
        fi.get("ebitda") for fi in financial_line_items if fi.get("ebitda") is not None
    ]

    # For EV calculation, let's get the most recent total_debt & cash
    debt_values = [
        fi.get("total_debt")
        for fi in financial_line_items
        if fi.get("total_debt") is not None
    ]
    cash_values = [
        fi.get("cash_and_equivalents")
        for fi in financial_line_items
        if fi.get("cash_and_equivalents") is not None
    ]
    recent_debt = debt_values[0] if debt_values else 0
    recent_cash = cash_values[0] if cash_values else 0

    enterprise_value = market_cap + recent_debt - recent_cash

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 15:
            pe_points = 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:
            pe_points = 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High or Very high P/E: {pe:.2f}")
        raw_score += pe_points
    else:
        details.append("No positive net income for P/E calculation")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 15:
            pfcf_points = 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < 25:
            pfcf_points = 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High/Very high P/FCF: {pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # 3) EV/EBIT
    recent_ebit = ebit_values[0] if ebit_values else None
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = enterprise_value / recent_ebit
        ev_ebit_points = 0
        if ev_ebit < 15:
            ev_ebit_points = 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < 25:
            ev_ebit_points = 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
        raw_score += ev_ebit_points
    else:
        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

    # 4) EV/EBITDA
    recent_ebitda = ebitda_values[0] if ebitda_values else None
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = enterprise_value / recent_ebitda
        ev_ebitda_points = 0
        if ev_ebitda < 10:
            ev_ebitda_points = 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < 18:
            ev_ebitda_points = 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
        raw_score += ev_ebitda_points
    else:
        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

    # We have up to 2 points for each of the 4 metrics => 8 raw points max
    # Scale raw_score to 0–10
    final_score = min(10, (raw_score / 8) * 10)

    return {"score": float(f"{final_score:.2f}"), "details": "; ".join(details)}
