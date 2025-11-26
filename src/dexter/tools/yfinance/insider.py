"""Insider trading tools backed by the yfinance API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from dexter.tools.yfinance.shared import get_ticker, to_python


def yf_get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Fetch insider trades from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.
        end_date: End date for filtering trades (YYYY-MM-DD).
        start_date: Optional start date for filtering trades (YYYY-MM-DD).
        limit: Maximum number of trades to return.
        api_key: Unused, kept for compatibility with signature.

    Returns:
        List of dictionaries matching the InsiderTrade schema.
    """
    ticker_obj = get_ticker(ticker)

    # yfinance returns a DataFrame for insider transactions
    try:
        trades_df = ticker_obj.insider_transactions
    except Exception:
        return []

    if trades_df is None or trades_df.empty:
        return []

    # Ensure we have a date column to filter on
    date_col = None
    for col in ["Start Date", "Date"]:
        if col in trades_df.columns:
            date_col = col
            break

    if not date_col:
        if isinstance(trades_df.index, pd.DatetimeIndex):
            trades_df = trades_df.reset_index()
            date_col = trades_df.columns[0]
        else:
            pass

    records = []

    # Parse filter dates
    try:
        end_dt = datetime.fromisoformat(end_date)
        start_dt = datetime.fromisoformat(start_date) if start_date else None
    except ValueError:
        end_dt = datetime.now()
        start_dt = None

    for idx, row in trades_df.iterrows():
        # Extract date
        trade_date = None
        if date_col and pd.notna(row.get(date_col)):
            val = row[date_col]
            if isinstance(val, (pd.Timestamp, datetime)):
                trade_date = val
            else:
                try:
                    trade_date = pd.to_datetime(val)
                except:
                    pass

        # Filter by date
        if trade_date:
            if trade_date > end_dt:
                continue
            if start_dt and trade_date < start_dt:
                continue

        # Extract values
        shares = to_python(row.get("Shares"))
        value = to_python(row.get("Value"))
        position = str(row.get("Position", ""))

        # Calculate price per share if possible
        price_per_share = None
        if (
            isinstance(shares, (int, float))
            and isinstance(value, (int, float))
            and shares != 0
        ):
            price_per_share = abs(value / shares)

        # Determine if board director
        is_director = "Director" in position if position else None

        record = {
            "ticker": ticker.upper(),
            "issuer": None,
            "name": str(row.get("Insider")) if pd.notna(row.get("Insider")) else None,
            "title": position if position else None,
            "is_board_director": is_director,
            "transaction_date": trade_date.isoformat() if trade_date else None,
            "transaction_shares": shares,
            "transaction_price_per_share": price_per_share,
            "transaction_value": value,
            "shares_owned_before_transaction": None,
            "shares_owned_after_transaction": to_python(row.get("Ownership")),
            "security_title": None,
            "filing_date": (
                trade_date.isoformat() if trade_date else ""
            ),  # Fallback to transaction date
        }

        records.append(record)

    # Sort by date descending
    records.sort(key=lambda x: x.get("transaction_date", "") or "", reverse=True)

    return records[:limit]
