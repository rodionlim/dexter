"""Price-related tools backed by the yfinance API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional

import pandas as pd
import numpy as np
from langchain.tools import tool

from dexter.tools.finance.prices import PriceSnapshotInput, PricesInput
from dexter.tools.yfinance.shared import get_ticker, to_python

_MINUTE_INTERVALS: dict[int, str] = {
    1: "1m",
    2: "2m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "60m",
    90: "90m",
}


def _resolve_history_request(
    interval: Literal["minute", "day", "week", "month", "year"],
    multiplier: int,
) -> tuple[str, Optional[str]]:
    """Map the abstract interval to yfinance's interval plus optional resample rule."""
    if interval == "minute":
        if multiplier not in _MINUTE_INTERVALS:
            raise ValueError(
                "yfinance supports minute intervals of 1, 2, 5, 15, 30, 60, or 90 minutes"
            )
        return _MINUTE_INTERVALS[multiplier], None

    if interval == "day":
        if multiplier == 1:
            return "1d", None
        if multiplier == 5:
            return "5d", None
        return "1d", f"{multiplier}D"

    if interval == "week":
        if multiplier == 1:
            return "1wk", None
        return "1d", f"{multiplier}W"

    if interval == "month":
        if multiplier == 1:
            return "1mo", None
        if multiplier == 3:
            return "3mo", None
        return "1d", f"{multiplier}M"

    if interval == "year":
        return "1mo", f"{multiplier}Y"

    raise ValueError(f"Unsupported interval: {interval}")


def _parse_iso_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD format."
        ) from exc


def _resample_prices(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adj Close": "last",
        "Volume": "sum",
        "Dividends": "sum",
        "Stock Splits": "sum",
    }
    resampled = frame.resample(rule).agg(agg)
    resampled = resampled.dropna(how="all")
    return resampled


def _history_to_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []

    frame = frame.dropna(how="all")
    if frame.empty:
        return []

    frame = frame.reset_index()
    time_key = frame.columns[0]
    records: list[dict] = []
    for _, row in frame.iterrows():
        timestamp = row[time_key]
        if isinstance(timestamp, pd.Timestamp):
            ts_value = timestamp.to_pydatetime().isoformat()
        elif isinstance(timestamp, datetime):
            ts_value = timestamp.isoformat()
        else:
            ts_value = str(timestamp)

        records.append(
            {
                "timestamp": ts_value,
                "open": to_python(row.get("Open")),
                "high": to_python(row.get("High")),
                "low": to_python(row.get("Low")),
                "close": to_python(row.get("Close")),
                "adj_close": to_python(row.get("Adj Close")),
                "volume": to_python(row.get("Volume")),
                "dividends": to_python(row.get("Dividends")),
                "stock_splits": to_python(row.get("Stock Splits")),
            }
        )
    return records


@tool(args_schema=PriceSnapshotInput)
def yf_get_price_snapshot(ticker: str) -> dict:
    """Fetch the latest Yahoo Finance quote snapshot (price, volume, market cap).

    Returns fast-moving fields such as last price, day range, previous close,
    and recent volume using yfinance's `fast_info` plus metadata fallbacks. Use
    this for real-time oriented prompts while operating in `yfinance` mode.
    """
    ticker_obj = get_ticker(ticker)
    snapshot: dict[str, Optional[float | str]] = {"ticker": ticker.upper()}

    fast_info = getattr(ticker_obj, "fast_info", None)
    if fast_info is not None:
        snapshot.update(
            {
                "currency": getattr(fast_info, "currency", None),
                "last_price": to_python(getattr(fast_info, "last_price", None)),
                "previous_close": to_python(getattr(fast_info, "previous_close", None)),
                "open": to_python(getattr(fast_info, "open", None)),
                "day_high": to_python(getattr(fast_info, "day_high", None)),
                "day_low": to_python(getattr(fast_info, "day_low", None)),
                "volume": to_python(getattr(fast_info, "last_volume", None)),
                "market_cap": to_python(getattr(fast_info, "market_cap", None)),
            }
        )

    info = getattr(ticker_obj, "info", {}) or {}
    snapshot.setdefault("currency", info.get("currency"))
    snapshot.setdefault("last_price", to_python(info.get("regularMarketPrice")))
    snapshot.setdefault(
        "previous_close", to_python(info.get("regularMarketPreviousClose"))
    )
    snapshot.setdefault("open", to_python(info.get("regularMarketOpen")))
    snapshot.setdefault("day_high", to_python(info.get("regularMarketDayHigh")))
    snapshot.setdefault("day_low", to_python(info.get("regularMarketDayLow")))
    snapshot.setdefault("volume", to_python(info.get("regularMarketVolume")))
    snapshot.setdefault("market_cap", to_python(info.get("marketCap")))

    market_time = info.get("regularMarketTime")
    if isinstance(market_time, (int, float)):
        snapshot["market_time"] = datetime.fromtimestamp(market_time).isoformat()

    return {
        "data_source": "yfinance",
        "snapshot": snapshot,
    }


@tool(args_schema=PricesInput)
def yf_get_prices(
    ticker: str,
    interval: Literal["minute", "day", "week", "month", "year"],
    interval_multiplier: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Download historical OHLCV bars from Yahoo Finance with optional resampling.

    Supports `minute`, `day`, `week`, `month`, and `year` intervals via the
    `interval`/`interval_multiplier` pair and respects `start_date`/`end_date`
    in ISO format. Use this when the agent needs price series but is configured
    to use the yfinance backend instead of FinancialDatasets.
    """
    ticker_obj = get_ticker(ticker)

    base_interval, resample_rule = _resolve_history_request(
        interval, interval_multiplier
    )
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)

    history = ticker_obj.history(
        start=start, end=end, interval=base_interval, auto_adjust=False
    )
    if resample_rule:
        history = _resample_prices(history, resample_rule)

    records = _history_to_records(history)
    return {
        "data_source": "yfinance",
        "ticker": ticker.upper(),
        "interval": interval,
        "interval_multiplier": interval_multiplier,
        "start_date": start_date,
        "end_date": end_date,
        "prices": records,
    }


@tool(args_schema=PricesInput)
def yf_get_price_performance(
    ticker: str,
    interval: Literal["minute", "day", "week", "month", "year"],
    interval_multiplier: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Calculate returns, drawdowns, range, and volatility for a ticker.

    **IMPORTANT**: Always prefer calling this function over yf_get_prices when raw
    price data is not required. This function returns computed performance metrics
    instead of raw OHLCV bars, significantly reducing input tokens and improving
    efficiency. Try to only call this function once, so pass in a broad enough date range.

    Returns:
    - Total return between start_date and end_date
    - Total annualized return (CAGR)
    - Periodic returns (1w, 1m, 3m, 6m, 1y, 3y, ytd) when data is available
    - 52-week high/low range
    - Maximum drawdown over the period
    - Annualized volatility
    - Current drawdown from peak
    - Start and End price

    Supports `minute`, `day`, `week`, `month`, and `year` intervals via the
    `interval`/`interval_multiplier` pair and respects `start_date`/`end_date`
    in ISO format.
    """
    ticker_obj = get_ticker(ticker)

    base_interval, resample_rule = _resolve_history_request(
        interval, interval_multiplier
    )
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)

    history = ticker_obj.history(
        start=start, end=end, interval=base_interval, auto_adjust=False
    )
    if resample_rule:
        history = _resample_prices(history, resample_rule)

    if history.empty:
        return {
            "data_source": "yfinance",
            "ticker": ticker.upper(),
            "interval": interval,
            "interval_multiplier": interval_multiplier,
            "start_date": start_date,
            "end_date": end_date,
            "error": "No price data available for the specified period",
        }

    # Use adjusted close for calculations
    prices = history["Adj Close"].dropna()
    if len(prices) == 0:
        return {
            "data_source": "yfinance",
            "ticker": ticker.upper(),
            "interval": interval,
            "interval_multiplier": interval_multiplier,
            "start_date": start_date,
            "end_date": end_date,
            "error": "No adjusted close prices available",
        }

    # Calculate total return (start to end)
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    total_return = (end_price - start_price) / start_price

    # Calculate total annualized return
    start_dt = prices.index[0]
    end_dt_for_calc = prices.index[-1]
    days_elapsed = (end_dt_for_calc - start_dt).days
    if days_elapsed > 0:
        years_elapsed = days_elapsed / 365.25
        total_annualized_return = to_python(
            ((1 + total_return) ** (1 / years_elapsed)) - 1
        )
    else:
        total_annualized_return = None

    total_return = to_python(total_return)

    # Calculate periodic returns
    periodic_returns = {}
    end_dt = prices.index[-1]

    # 1 week return
    one_week_ago = end_dt - timedelta(days=7)
    week_prices = prices[prices.index >= one_week_ago]
    if len(week_prices) >= 2:
        periodic_returns["1w"] = to_python(
            (week_prices.iloc[-1] - week_prices.iloc[0]) / week_prices.iloc[0]
        )

    # 1 month return
    one_month_ago = end_dt - timedelta(days=30)
    month_prices = prices[prices.index >= one_month_ago]
    if len(month_prices) >= 2:
        periodic_returns["1m"] = to_python(
            (month_prices.iloc[-1] - month_prices.iloc[0]) / month_prices.iloc[0]
        )

    # 3 month return
    three_months_ago = end_dt - timedelta(days=90)
    three_month_prices = prices[prices.index >= three_months_ago]
    if len(three_month_prices) >= 2:
        periodic_returns["3m"] = to_python(
            (three_month_prices.iloc[-1] - three_month_prices.iloc[0])
            / three_month_prices.iloc[0]
        )

    # 6 month return
    six_months_ago = end_dt - timedelta(days=180)
    six_month_prices = prices[prices.index >= six_months_ago]
    if len(six_month_prices) >= 2:
        periodic_returns["6m"] = to_python(
            (six_month_prices.iloc[-1] - six_month_prices.iloc[0])
            / six_month_prices.iloc[0]
        )

    # 1 year return
    one_year_ago = end_dt - timedelta(days=365)
    one_year_prices = prices[prices.index >= one_year_ago]
    if len(one_year_prices) >= 2:
        periodic_returns["1y"] = to_python(
            (one_year_prices.iloc[-1] - one_year_prices.iloc[0])
            / one_year_prices.iloc[0]
        )

    # 3 year return
    three_years_ago = end_dt - timedelta(days=1095)
    three_year_prices = prices[prices.index >= three_years_ago]
    if len(three_year_prices) >= 2:
        periodic_returns["3y"] = to_python(
            (three_year_prices.iloc[-1] - three_year_prices.iloc[0])
            / three_year_prices.iloc[0]
        )

    # Year-to-date return - use pd.Timestamp to preserve timezone
    ytd_start = pd.Timestamp(year=end_dt.year, month=1, day=1, tz=end_dt.tz)
    ytd_prices = prices[prices.index >= ytd_start]
    if len(ytd_prices) >= 2:
        periodic_returns["ytd"] = to_python(
            (ytd_prices.iloc[-1] - ytd_prices.iloc[0]) / ytd_prices.iloc[0]
        )

    # Calculate 52-week range
    fifty_two_weeks_ago = end_dt - timedelta(days=365)
    week_52_prices = prices[prices.index >= fifty_two_weeks_ago]
    if len(week_52_prices) > 0:
        week_52_high = to_python(week_52_prices.max())
        week_52_low = to_python(week_52_prices.min())
    else:
        week_52_high = None
        week_52_low = None

    # Calculate drawdowns
    cumulative_max = prices.expanding().max()
    drawdowns = (prices - cumulative_max) / cumulative_max
    max_drawdown = to_python(drawdowns.min())
    current_drawdown = to_python(drawdowns.iloc[-1])

    # Calculate annualized volatility
    if len(prices) > 1:
        returns = prices.pct_change().dropna()

        # Determine annualization factor based on interval
        if interval == "minute":
            # Assume 6.5 trading hours per day, 252 trading days per year
            periods_per_year = 252 * 6.5 * 60 / interval_multiplier
        elif interval == "day":
            periods_per_year = 252 / interval_multiplier
        elif interval == "week":
            periods_per_year = 52 / interval_multiplier
        elif interval == "month":
            periods_per_year = 12 / interval_multiplier
        elif interval == "year":
            periods_per_year = 1 / interval_multiplier
        else:
            periods_per_year = 252

        volatility = to_python(returns.std() * np.sqrt(periods_per_year))
    else:
        volatility = None

    return {
        "data_source": "yfinance",
        "ticker": ticker.upper(),
        "interval": interval,
        "interval_multiplier": interval_multiplier,
        "start_date": start_date,
        "end_date": end_date,
        "start_price": start_price,
        "end_price": end_price,
        "performance": {
            "total_return": total_return,
            "total_annualized_return": total_annualized_return,
            "periodic_returns": periodic_returns,
            "52_week_high": week_52_high,
            "52_week_low": week_52_low,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "annualized_volatility": volatility,
        },
    }
